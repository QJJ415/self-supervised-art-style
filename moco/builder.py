# References:
# Moco-v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

from pdb import set_trace

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from utils.utils import nt_xent_debiased
from utils.utils import VIC_loss

from utils.attn_distill import overlap_clip, cross_img_attn_dist_loss


def predictor_temp_scheduler(epoch):
    if epoch < 100:
        return - (30 - 1) / 100 * epoch + 30
    else:
        return 1

class Conditioned_Mlp(nn.Module):
    def __init__(self, dim, mlps):
        super(Conditioned_Mlp, self).__init__()

        self.mlps = mlps
        self.gate = nn.Linear(dim * 2, 4)

    def forward(self, q, k, temp=1.0):
        q_pred = []
        for mlp in self.mlps:
            q_pred.append(mlp(q))
        q_pred = torch.stack(q_pred, dim=1)

        # print("temp is {}".format(temp))
        gate = self.gate(torch.cat([q, k], dim=-1)) / temp
        gate = F.softmax(gate, dim=-1)

        return (q_pred * gate.unsqueeze(-1)).sum(1)


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, simclr_version=False, VIC_version=False,
                 return_features=False,
                 return_representation=False, conditioned_predictor=False, conditioned_predictor_temp=False,
                 attn_distill=False, attn_distill_cross_view=False, attn_distill_layers_num=1, cmae=False,
                 mae_aug=False, num_heads=16,
                 mlp_ratio=4., depth=12, norm_layer=nn.LayerNorm, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768,
                 T1=0.1, T2=0.05, type='ascl', nn_num=1, mem_size=4096, in_channels=768, channels=256, coeff=0.5):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.simclr_version = simclr_version
        self.VIC_version = VIC_version

        self.ID = [9,10,11,12]
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(len(self.ID))])
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.num_layers = 3
        self.hidden_dim = 768
        self.projection_layers = nn.ModuleList([
            nn.Linear(embed_dim, self.hidden_dim) for _ in range(self.num_layers)
        ])
        # SOFT
        self.coeff = coeff
        self.T1 = T1
        self.T2 = T2
        self.K = mem_size
        self.max_entropy = np.log(self.K)
        self.nn_num = nn_num
        self.type = type
        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('labels', -1 * torch.ones(self.K).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # gram 矩阵
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.mse = nn.MSELoss()
        self.q = nn.Linear(in_channels,in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.scale = self.head_dim ** -0.5  # 缩放因子
        # 通过一个可学习的线性变换对通道进行变换
        self.proj = nn.Linear(in_channels, in_channels, bias=False)
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        if (not self.simclr_version) and (not self.VIC_version):
            self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        if (not simclr_version) and (not self.VIC_version):
            for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        self.return_features = return_features
        self.return_representation = return_representation
        self.conditioned_predictor = conditioned_predictor
        self.conditioned_predictor_temp = conditioned_predictor_temp

        if self.return_representation:
            self.base_encoder.head = nn.Identity()
            self.momentum_encoder.head = nn.Identity()
            self.predictor = nn.Identity()

        if self.conditioned_predictor:
            self.predictor = Conditioned_Mlp(dim, nn.ModuleList([self._build_mlp(2, dim, mlp_dim, dim)
                                                                 for _ in range(4)]))

        self.attn_distill = attn_distill
        self.attn_distill_cross_view = attn_distill_cross_view
        self.attn_distill_layers_num = attn_distill_layers_num
        self.cmae = cmae
        self.mae_aug = mae_aug
        if self.conditioned_predictor_temp:
            self.predictor_temp_scheduler = predictor_temp_scheduler
        else:
            self.predictor_temp_scheduler = None

    def gram_matrix(self, features):
        """
                   features: Tensor [B, D, N]
                   返回: Gram 矩阵 [B, D,D]
                   """
        B, N, D = features.size()
        x_t = features.transpose(1, 2)  # [B, D, N]
        gram = torch.bmm(x_t, features)  # [B, D, D]

        gram = gram / features.shape[1]
        return gram

    def gram_loss(self, gram1, gram2):
        gram1 = gram1.reshape(-1)
        gram2 = gram2.reshape(-1)

        loss = self.mse(gram1, gram2)
        return loss


    def learn_gram(self, x):
        """
                x: [B, N, D]
                return: Softmax Gram matrix [B, N, N]
                """
        Q = self.q(x)  # [B, N, D]
        # print("Q",Q.shape)
        K = self.k(x)  # [B, N, D]

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, N, N]
        gram = torch.softmax(attn_scores, dim=-1)
        #x_proj = self.proj(x)  # [B, N, D]
        #x_proj_T = x_proj.transpose(1, 2)  # [B, D, N]
        #gram = torch.bmm(x_proj_T, x_proj) / x.shape[1]  # [B, D, D]
        #print("gram",gram.shape)
        return gram

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def simclr_loss(self, q1, q2):
        # normalize
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        q1 = concat_all_gather_wGrad(q1)
        q2 = concat_all_gather_wGrad(q2)

        return nt_xent_debiased(q1, features2=q2, t=self.T) * torch.distributed.get_world_size()

    def distill_attn(self, q1_attns, k1_attns):
        loss_all = 0
        for cnt_layer, (q1_attn, k1_attn) in enumerate(zip(q1_attns, k1_attns)):
            # distill the attn for last k layers
            if cnt_layer >= len(q1_attns) - self.attn_distill_layers_num:
                loss = - (k1_attn * torch.log(q1_attn)).sum(-1).mean()
                loss_all += loss

        return loss_all / self.attn_distill_layers_num

    def cross_distill_attn(self, attns1, attns2, bbox1, bbox2, h_attn, w_attn):
        loss_all = 0
        for cnt_layer, (attn1, attn2) in enumerate(zip(attns1, attns2)):
            # distill the attn for last k layers
            if cnt_layer >= len(attns1) - self.attn_distill_layers_num:
                loss = cross_img_attn_dist_loss(attn1, attn2, bbox1, bbox2, h_attn, w_attn)
                loss_all += loss

        return loss_all / self.attn_distill_layers_num

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def random_masking_gene_id(self, N, L, device, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        return ids_restore, ids_keep

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        self.labels[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def _forward_moco(self, im_q, im_k, targets, m):
        # update the key encoder
        with torch.no_grad():
            self._update_momentum_encoder(m)
        q = self.base_encoder(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        batch_size = q.shape[0]
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            k = self.momentum_encoder(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # pseudo logits: NxK
            logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
            sim_k_ktarget = torch.zeros(batch_size, device='cuda').unsqueeze(-1)
            sim_k = torch.cat([sim_k_ktarget, logits_pd], dim=1)
            logits_k = sim_k / self.T2

            labels = torch.zeros(logits_pd.size(0), logits_pd.size(1) + 1).cuda()
            if self.type == 'ascl':
                labels[:, 0] = 1.0
                pseudo_labels = F.softmax(logits_pd, 1)
                log_pseudo_labels = F.log_softmax(logits_pd, 1)
                entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
                c = 1 - entropy / self.max_entropy
                pseudo_labels = self.nn_num * c * pseudo_labels  # num of neighbors * uncertainty * pseudo_labels
                pseudo_labels = torch.minimum(pseudo_labels,
                                              torch.tensor(1).to(pseudo_labels.device))  # upper thresholded by 1
                labels[:, 1:] = pseudo_labels  # summation <= c*K <= K
            else:  # no extra neighbors [moco]
                labels[:, 0] = 1.0

        # label normalization
        labels = labels / labels.sum(dim=1, keepdim=True)
        # print(labels.dtype)  # 检查labels的类型
        # forward pass
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # sim_q

        logits_q = logits / self.T1

        label = torch.zeros(
            batch_size, dtype=torch.long, device='cuda')
        mask = nn.functional.one_hot(
            label, 1 + self.queue.shape[1])
        # print("m", mask.shape) m torch.Size([4, 4097])
        prob_k = nn.functional.softmax(logits_k, dim=1)
        prob_q = nn.functional.normalize(
            self.coeff * mask + (1 - self.coeff) * prob_k, p=1, dim=1)

        loss = - \
            torch.sum(prob_q * nn.functional.log_softmax(logits_q,
                                                         dim=1), dim=1).mean(dim=0)
        # loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()

        self._dequeue_and_enqueue(k, targets)
        return loss

    def gram_features(self, model, x):
        x = self.patch_embed(x)
        # print("pos",x.shape)pos[4,196, 768])
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = model.pos_drop(x + model.pos_embed)
        # 仅经过 encoder blocks（不经过 classifier）
        for block in self.blocks:
            x = block(x)
        encoder_output = model.norm(x)  # shape: [B, 197, 768]
        # print(encoder_output.shape)
        # print(x.shape)torch.Size([4, 197, 768])

        return encoder_output

    def extract_intermediate_features(self, model, x):
        x = self.patch_embed(x)
        latent = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            # print("x",x.shape)torch.Size([8, 196, 768])
            if i in self.ID:
                latent.append(self.norm[self.ID.index(i)](x))
        # print(type(features))
        return latent
    def attention(self, model, x):
        x = self.patch_embed(x)
        latent = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i in self.ID:
                latent.append(self.norm[self.ID.index(i)](x))
        last_four_tensor = torch.stack(latent)  # 将后四层的特征堆叠为一个张量，形状为 [4, B, N, C]
        num_layers, B, N, C = last_four_tensor.shape
        Q = self.query(last_four_tensor)  # (batch_size, num_layers, feature_dim)
        K = self.key(last_four_tensor)  # (batch_size, num_layers, feature_dim)
        V = self.value(last_four_tensor)  # (batch_size, num_layers, feature_dim)
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, num_layers, num_layers)
        attn_weights = F.softmax(attn_scores, dim=-1)  # 每层的权重 (batch_size, num_layers, num_layers)
        # 加权求和得到融合特征
        weighted_features = torch.matmul(attn_weights, V)  # (batch_size, num_layers, feature_dim)
        return weighted_features
    def multi_layer_contrastive_loss(self, layer_features_q, layer_features_k):
        loss = 0
        num_layers = len(layer_features_q)

        for i in range(num_layers):
            cls_token_q = layer_features_q[i][:, 0, :]  # [16, 768]
            cls_token_k = layer_features_k[i][:, 0, :]  # [16, 768]

            cls_token_q = self.projection_layers[i](cls_token_q)  # 投影查询特征
            cls_token_k = self.projection_layers[i](cls_token_k)  # 投影键特征
            q = nn.functional.normalize(cls_token_q, dim=-1)

            k = nn.functional.normalize(cls_token_k, dim=-1)
            k = concat_all_gather(k)
            logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
            N = logits.shape[0]  # batch size per GPU
            labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
            loss += nn.CrossEntropyLoss()(logits, labels)
        return loss / num_layers  # 对所有层的损失进行平均

    def forward(self, x1, x2, targets, m, epoch=-1, record_feature=False, bboxes=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        # qqq = self.base_encoder(x1)
        # print("x1", x1.shape) # x1 torch.Size([8, 3, 224, 224])BCHW
        # print("q1 = self.predictor(self.base_encoder(x1))",qqq.shape)
        if self.mae_aug:
            _, ids_keep_1 = self.random_masking_gene_id(x1.shape[0], self.base_encoder.patch_embed.num_patches,
                                                        x1.device, mask_ratio=0.75)
            _, ids_keep_2 = self.random_masking_gene_id(x2.shape[0], self.base_encoder.patch_embed.num_patches,
                                                        x1.device, mask_ratio=0.75)
        else:
            ids_keep_1, ids_keep_2 = None, None

        # print("x1.shape is {}".format(x1.shape))
        if self.cmae:
            assert not self.mae_aug
            assert not self.simclr_version
            assert not self.VIC_version
            assert not self.conditioned_predictor
            assert not self.attn_distill
            assert not self.return_features

            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder
                # compute momentum features as targets
                k2 = self.momentum_encoder(x2)

            # mask augmentation for base encoder
            q1_feature = self.base_encoder.patch_embed(x1)
            q1_feature = q1_feature + self.base_encoder.pos_embed[:, 1:, :]
            q1_feature, mask, ids_restore, ids_keep = self.random_masking(q1_feature, 0.75)

            # append cls token
            cls_token = self.base_encoder.cls_token + self.base_encoder.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(q1_feature.shape[0], -1, -1)
            q1_feature = torch.cat((cls_tokens, q1_feature), dim=1)

            for cnt, blk in enumerate(self.base_encoder.blocks):
                q1_feature = blk(q1_feature)

            q1 = self.predictor(self.base_encoder.head(q1_feature[:, 0]))

            return self.contrastive_loss(q1, k2)

        if (not self.simclr_version) and (not self.VIC_version):
            with torch.no_grad():  # no gradient
                self._update_momentum_encoder(m)  # update the momentum encoder

            if self.attn_distill:
                assert not self.mae_aug
                # compute momentum features as targets
                k1, k1_attns = self.momentum_encoder(x1, return_attn=True)
                k2, k2_attns = self.momentum_encoder(x2, return_attn=True)
            else:
                # compute momentum features as targets
                 k1 = self.momentum_encoder(x1, mask_ids_keep=ids_keep_1)
                 k2 = self.momentum_encoder(x2, mask_ids_keep=ids_keep_2)
        

        if not self.conditioned_predictor:
            features = {}

            if self.attn_distill:
                assert not self.mae_aug
                assert not self.VIC_version

                q1_feat, q1_attns = self.base_encoder(x1, return_attn=True, record_feat=record_feature)
                features["x1"] = self.base_encoder.recorded_feature
                self.base_encoder.recorded_feature = None
                q2_feat, q2_attns = self.base_encoder(x2, return_attn=True, record_feat=record_feature)
                features["x2"] = self.base_encoder.recorded_feature
                self.base_encoder.recorded_feature = None
                self.features = features

                q1 = self.predictor(q1_feat)
                q2 = self.predictor(q2_feat)
            else:
                if self.VIC_version:
                    assert not self.mae_aug
                    q1 = self.base_encoder(x1, record_feat=record_feature)
                    features["x1"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    q2 = self.base_encoder(x2, record_feat=record_feature)
                    features["x2"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    self.features = features
                else:
                    # 执行这里
                    q1 = self.predictor(self.base_encoder(x1, record_feat=record_feature, mask_ids_keep=ids_keep_1))
                  
                    features["x1"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    q2 = self.predictor(self.base_encoder(x2, record_feat=record_feature, mask_ids_keep=ids_keep_2))
                    features["x2"] = self.base_encoder.recorded_feature
                    self.base_encoder.recorded_feature = None
                    self.features = features
        else:
            assert not self.mae_aug
            assert not self.attn_distill
            assert not record_feature
            predictor_temp = 1 if self.predictor_temp_scheduler is None else self.predictor_temp_scheduler(epoch)
            q1 = self.predictor(self.base_encoder(x1), k2, predictor_temp)
            q2 = self.predictor(self.base_encoder(x2), k1, predictor_temp)

        if self.simclr_version:
            assert not self.mae_aug
            assert not self.return_features
            return self.simclr_loss(q1, q2)

        if self.VIC_version:
            assert not self.return_features
            return VIC_loss(q1, q2)

        if self.return_features:
            assert not self.attn_distill
            q1 = concat_all_gather_wGrad(q1.contiguous())
            q2 = concat_all_gather_wGrad(q2.contiguous())
            k1 = concat_all_gather_wGrad(k1.contiguous())
            k2 = concat_all_gather_wGrad(k2.contiguous())
            return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), q1, q2, k1, k2
        else:
            if self.attn_distill:
                if not self.attn_distill_cross_view:
                    return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), \
                           self.distill_attn(q1_attns, k1_attns) + self.distill_attn(q2_attns, k2_attns)
                else:
                    bbox1, bbox2 = bboxes
                    # verify the overlap region clip code
                    # overlap_clip(x1, x2, bbox1, bbox2)
                    h_attn = self.base_encoder.patch_embed.grid_size[0]
                    w_attn = self.base_encoder.patch_embed.grid_size[1]
                    return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), \
                           self.cross_distill_attn(q1_attns, k2_attns, bbox1, bbox2, h_attn, w_attn) + \
                           self.cross_distill_attn(q2_attns, k1_attns, bbox2, bbox1, h_attn, w_attn)
            else:

                # 获取 query 和 key 的中间层特征
                intermediate_features_q = self.attention(self.base_encoder, x1)
                intermediate_features_k = self.attention(self.momentum_encoder, x2)
                #软对比损失
                soft_loss = self._forward_moco(x1, x2, targets, m) + self._forward_moco(x2, x1, targets, m)
                # 计算多层次对比损失
                contrastive_loss1 = self.multi_layer_contrastive_loss(intermediate_features_q, intermediate_features_k)

                gram_feature1 = self.gram_features(self.base_encoder, x1)
                gram_feature2 = self.gram_features(self.base_encoder, x2)
                # 对比风格向量
                gram1 = self.learn_gram(gram_feature1)
                gram2 = self.learn_gram(gram_feature2)
                 #计算gram矩阵损失
                gram_loss = self.gram_loss(gram1, gram2)
                return gram_loss + soft_loss+contrastive_loss1


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc  # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head
        if (not self.simclr_version) and (not self.VIC_version):
            del self.momentum_encoder.head  # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        if (not self.simclr_version) and (not self.VIC_version):
            self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        if not self.VIC_version:
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def concat_all_gather_wGrad(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    tensors_gather[torch.distributed.get_rank()] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output
