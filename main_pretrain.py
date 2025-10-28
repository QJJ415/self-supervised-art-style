
# References:
# Moco-v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
from pdb import set_trace
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch import nn
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import moco.builder
import moco.loader
import moco.optimizer
from byol_pytorch import NetWrapper
from models import vits
from collections import OrderedDict
from models.vision_transformer import DINOHead
import self_utils
from utils.utils import logger, sync_weights
from utils.init_datasets import init_datasets
from utils.pretrain import pretrain_transform
import utils_mae.lr_decay as lrd
import numpy as np
from utils.pretrain import evaluate_VIC
from utils.l1_weight_regu import L1Regularizer

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_large', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

gate_names = ['', 'vit_gate_small', 'vit_gate_base', 'vit_gate_large']

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')

parser.add_argument('--experiment', type=str, default='')
parser.add_argument('--save_dir', type=str, default="./checkpoints")
parser.add_argument('--data', default="", metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='AVA')  # 在vision_transformer里面改了img_size=112
parser.add_argument('--customSplit', type=str, default='')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--local_rank', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--save_freq', default=2000, type=int, help='The freq for saving model.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--no_save_last', action='store_true')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=40, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# simclr version
parser.add_argument('--simclr_version', action='store_true', help='simclr version')
parser.add_argument('--VIC_version', action='store_true', help='VIC version')

# evaluate pretrain performance
parser.add_argument('--evaluate_pretrain', action='store_true', help='if evaluate pretrain performance')
parser.add_argument('--evaluate_VIC', action='store_true', help='if evaluate VIC')
parser.add_argument('--evaluate_pretrain_representation', action='store_true',
                    help='if evaluate pretrain representation')

# grafting
parser.add_argument('--graft_pretrained', default="", type=str,
                    help='the pretrain path for grafting')
parser.add_argument('--end_pretrained', default="", type=str, help='the pretrain path for the grafting part')
parser.add_argument('--fixTo', default=-1, type=int, help='fix previous layers for grafting')
parser.add_argument('--graft_CL_end', action="store_true", help='if use CL end for grafting')
parser.add_argument('--scratch_end', action="store_true", help='if random initalize the last layers')
parser.add_argument('--l1_dist_w', default=-1, type=float)
parser.add_argument('--l1_dist_to_block', default=3, type=int,
                    help="apply l1 regu to blocks smaller than this number")

# save local crops
parser.add_argument('--local_crops_number', default=0, type=int, help='the local crops number')

# conditioned predictor
parser.add_argument('--conditioned_predictor', action='store_true', help='if employ conditioned_predictor')
parser.add_argument('--conditioned_predictor_temp', action='store_true', help='if employ conditioned_predictor_temp')

# lr schedule
parser.add_argument('--layer_wise_decay_ratio', default=-1, type=float, help='if employs the layer_wise_decay')
parser.add_argument('--lr_layer_wise', default="1.5e-5,1.5e-5,1.5e-4", type=str,
                    help='if directly assign lr to different layers')

# moco dropping augmentation
parser.add_argument('--mae_aug', action="store_true", help='if applying MAE augmentation', default=True)

# self distillation
parser.add_argument('--out_dim', default=4096, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
parser.add_argument('--norm_last_layer', default=True, type=self_utils.bool_flag,
                    help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
parser.add_argument('--use_bn_in_head', default=False, type=self_utils.bool_flag,
                    help="Whether to use batch normalizations in projection head (Default: False)")
parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
       parameter for teacher update. The value is increased to 1 during training with cosine schedule.
       We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
# Temperature teacher parameters
parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                    help="""Initial value for the teacher temperature: 0.04 works well in most cases.
    Try decreasing it if the training loss does not decrease.""")
parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
    of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
    starting with the default value of 0.04 and increase this slightly if needed.""")
parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                    help='Number of warmup epochs for the teacher temperature (Default: 30).')
parser.add_argument('--local_crop_number', type=int, default=0, help="""Number of small  
       local views to generate. Set this parameter to 0 to disable multi-crop training.
       When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)  # 不使用multi-crop
parser.add_argument('--weight_decay2', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
#
parser.add_argument('--mlp_hidden', default=4096, type=int, help='mlp hidden dimension')

def main():

    print(torch.cuda.is_available())
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.local_rank == -1:
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    logName = "log.txt"
    save_dir = os.path.join(args.save_dir, args.experiment)
    if not os.path.exists(save_dir):
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir, log_name=logName)

    # Simply call main_worker function
    main_worker(args.local_rank, args, log)


def fixBackBone(model, fixTo):
    if fixTo > 0:
        # fix the embedding
        model.module.base_encoder.pos_embed.requires_grad = False
        model.module.base_encoder.cls_token.requires_grad = False

        for param in model.module.base_encoder.patch_embed.parameters():
            param.requires_grad = False
        model.module.base_encoder.patch_embed.eval()

        for layer in model.module.base_encoder.blocks[:fixTo]:
            for param in layer.parameters():
                param.requires_grad = False
            layer.eval()


def merge_backbone(model_graft, model_end, fixTo, end_block, log):
    state_dict_new = OrderedDict()
    if fixTo > 0:
        # delete the state dict of shallow layer in model_end
        if model_end is not None:
            if list(model_end.keys())[0].startswith('module.'):
                model_end = {k[7:]: v for k, v in model_end.items()}

            if sorted(list(model_end.keys()))[0].startswith('base_encoder.'):
                linear_keyword = "head"
                for k in list(model_end.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                        if "_aux" in k:
                            # print("skip k is {}".format(k))
                            continue
                        # remove prefix
                        model_end[k[len("base_encoder."):]] = model_end[k]
                    # delete renamed or unused k
                    del model_end[k]

            del model_end['pos_embed']
            del model_end['cls_token']
            del model_end['patch_embed.proj.weight']
            del model_end['patch_embed.proj.bias']

            to_delete = []
            for cnt_layer in range(fixTo):
                for key in model_end:
                    if "blocks.{}.".format(cnt_layer) in key:
                        to_delete.append(key)

            log.info("Keys deleted in the end model: {}".format(to_delete))
            for key in to_delete:
                del model_end[key]

            # merge
            for key, item in model_end.items():
                state_dict_new[key] = item

        # delete the state dict of deep layer in model_end
        to_delete = ["norm.weight", "norm.bias"]
        for cnt_layer in range(fixTo, end_block):
            for key in model_graft:
                if "blocks.{}.".format(cnt_layer) in key:
                    to_delete.append(key)

        log.info("Keys deleted in the graft model: {}".format(to_delete))
        for key in to_delete:
            del model_graft[key]

        # merge
        for key, item in model_graft.items():
            if key in state_dict_new:
                raise ValueError("Overlap keys: {}".format(key))
            state_dict_new[key] = item

        return state_dict_new
    else:
        return model_graft


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir="checkpoints", only_best=False):
    if not only_best:
        torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))



class teacher_mid_loss(nn.Module):
        def __init__(self, student, teacher,  embed_dim, args):
            super().__init__()
            # self.save_hyperparameters()
            self.ratio = 0
            self.st_inter = True # true
            self.t_inter = False
            self.criterion = nn.CrossEntropyLoss()
            # teacher.load_state_dict(student.state_dict())

            self.student = NetWrapper(student, embed_dim, args, prediction=False, intermediate=self.st_inter)
            self.teacher = NetWrapper(teacher, embed_dim, args, prediction=True, intermediate=self.t_inter)

            if self.st_inter != self.t_inter:
                # 这句
                self.student.projector.load_state_dict(self.teacher.projector.state_dict(),strict=False)
            else:
                self.student.projector.load_state_dict(self.teacher.projector.state_dict())

            # print(self.teacher.projector)
            # print(self.student)
            # print("teacher", self.teacher)
        def info_nce_loss_layer(self, s_features, t_features, d=None):
            batch_size = s_features.shape[0]
            if d is not None:
                depth = d
            else:
                depth = 11 if self.t_inter else 1

            batch_size /= depth
            # print("b", batch_size) # 4
            labels = torch.cat([(torch.arange(batch_size, dtype=torch.long) + int(batch_size * torch.distributed.get_rank())) for _ in range(depth)], dim=0).cuda()
            labels = labels.float()
            # print(labels)

            s_features = F.normalize(s_features, dim=1)

            t_features = F.normalize(t_features, dim=1)

            similarity_matrix = torch.matmul(s_features, t_features.T)

            tau = 0.2

            logits = similarity_matrix / tau
            logits = logits[:, 0]

            loss = self.criterion(logits, labels)
            return 2 * tau * loss

        def forward(self, x1, x2):
            batch_size = x1.shape[0]
            # print("b", batch_size) # 8
            # x1 = x1.cuda()
            # x2 = x2.cuda()
            teacher_output1 = self.teacher(x1)
            student_output1 = self.student(x2)
            teacher_output2 = self.teacher(x2)
            student_output2 = self.student(x1)

            """manual update for predictor"""
            student_detached1 = self.student.predict(student_output1.detach())
            student_detached2 = self.student.predict(student_output2.detach())


            """compute loss for vit and projector (update predictor slightly)"""
            student_output1 = self.student.predict(student_output1)
            student_output2 = self.student.predict(student_output2)
            # print(student_output1.shape)  # 打印张量的形状 torch.Size([4, 4096])
            student_mid1, student_output1 = torch.split(student_output1, [batch_size * 11, batch_size], dim=0)
            # print("ss", student_mid1.shape, student_output1.shape) # torch.Size([44, 4096]) torch.Size([4, 4096])
            student_mid2, student_output2 = torch.split(student_output2, [batch_size * 11, batch_size], dim=0)

            loss_mid = self.info_nce_loss_layer(student_mid1, teacher_output1) + self.info_nce_loss_layer(student_mid2, teacher_output2)

            return loss_mid


def main_worker(local_rank, args, log):
    torch.cuda.empty_cache()
    args.local_rank = local_rank
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '56285'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    dist.init_process_group(backend='gloo', init_method="env://?use_libuv=false", rank=0,
                            world_size=os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ else 1)
    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and args.local_rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass
        log.local_rank = 1

    log.info(str(args))

    if args.local_rank is not None:
        log.info("Use GPU: {} for training".format(args.local_rank))

    if args.distributed:
        # print("os.environ are {}".format(str(os.environ)))
        # print("os.environ AZUREML_CR_NODE_RANK is {}".format(os.environ['AZUREML_CR_NODE_RANK']))
        # print("os.environ AZ_BATCH_MASTER_NODE is {}".format(os.environ['AZ_BATCH_MASTER_NODE']))

        if 'AZ_BATCH_MASTER_NODE' in os.environ:
            NODE_RANK = os.environ['AZUREML_CR_NODE_RANK'] \
                if 'AZUREML_CR_NODE_RANK' in os.environ else 0
            NODE_RANK = int(NODE_RANK)
            MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') \
                if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
            MASTER_PORT = int(MASTER_PORT)
            DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
            ngpus_per_node = torch.cuda.device_count()
            args.rank = NODE_RANK * ngpus_per_node + args.local_rank
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=DIST_URL,
                                                 world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method="env://", rank=0, world_size=args.world_size)
        torch.distributed.barrier()

    # args.rank = torch.distributed.get_rank()
    # log.local_rank = args.rank
    # create model
    log.info("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.builder_soft_gram.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, return_features=args.evaluate_pretrain,
            VIC_version=args.VIC_version,
            return_representation=args.evaluate_pretrain_representation,
            conditioned_predictor=args.conditioned_predictor,
            conditioned_predictor_temp=args.conditioned_predictor_temp,
            mae_aug=args.mae_aug)
        student = vits.__dict__[args.arch](
            drop_path_rate=args.drop_path_rate, num_classes=768  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](num_classes=768)
        embed_dim = student.embed_dim
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    log.info(str(model))
    # move networks to gpu
    student, teacher = student.cuda(args.local_rank), teacher.cuda(args.local_rank)

    # move networks to gpu
    student, teacher = student.cuda(args.local_rank), teacher.cuda(args.local_rank)
    # synchronize batch norms (if any)
    if self_utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    # print(student)
    # ============ preparing loss ... ============
    teachermid_loss = teacher_mid_loss(student, teacher, embed_dim, args)

    if not torch.cuda.is_available():
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            log.info("the world size is {}".format(torch.distributed.get_world_size()))
            args.batch_size = int(args.batch_size / torch.distributed.get_world_size())
            log.info("the batch size each gpu is {}".format(args.batch_size))
            args.workers = args.workers
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.local_rank is not None:
        print("gpu")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # torch.cuda.set_device(device)
        # model = model.cuda()
        torch.cuda.set_device(args.local_rank)
        model = model.cuda(args.local_rank)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    if args.layer_wise_decay_ratio > 0:
        assert args.lr_layer_wise == ""
        parameters = lrd.param_groups_lrd_moco(model.module, args.weight_decay, no_weight_decay_list=[],
                                               layer_decay=args.layer_wise_decay_ratio)
    elif args.lr_layer_wise != "":
        parameters = lrd.param_groups_lrd_moco(model, args.weight_decay, no_weight_decay_list=[],
                                               lr_layer_wise=args.lr_layer_wise)
    else:
        parameters = [param for param in model.parameters() if param.requires_grad]

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(parameters, args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, args.lr,
                                      weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer of {} is not supported.".format(args.optimizer))

    params_student = self_utils.get_params_groups(student)
    optimizer_vit = torch.optim.AdamW(params_student)  # to use with ViTs

    scaler = torch.cuda.amp.GradScaler()

    if args.l1_dist_w > 0:
        assert args.graft_pretrained != ""

    l1_regularizer = None
    if args.graft_pretrained != "":
        checkpoint_graft = torch.load(args.graft_pretrained, map_location="cpu")
        if 'state_dict' in checkpoint_graft:
            state_dict = checkpoint_graft['state_dict']
        elif 'model' in checkpoint_graft:
            state_dict = checkpoint_graft['model']
        elif 'module' in checkpoint_graft:
            state_dict = checkpoint_graft['module']
        else:
            state_dict = checkpoint_graft

        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # for moco v3 pre-trained keys
        if sorted(list(state_dict.keys()))[0].startswith('base_encoder.'):
            linear_keyword = "head"
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                    if "_aux" in k:
                        # print("skip k is {}".format(k))
                        continue
                    # remove prefix
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        if args.graft_CL_end or args.scratch_end:
            checkpoint_end = None
            if args.graft_CL_end:
                checkpoint_end = torch.load(args.end_pretrained, map_location="cpu")
                checkpoint_end = checkpoint_end['state_dict']

            state_dict = merge_backbone(state_dict, checkpoint_end, args.fixTo, len(model.module.base_encoder.blocks),
                                        log)

        msg_base = model.base_encoder.load_state_dict(state_dict, strict=False)
        print("msg_base is {}".format(msg_base))

        if (not args.simclr_version) and (not args.VIC_version):
            msg_momentum = model.momentum_encoder.load_state_dict(state_dict, strict=False)
            print("msg_momentum is {}".format(msg_momentum))

        if args.l1_dist_w > 0:
            # record the init weight
            l1_regularizer = L1Regularizer(model.module.base_encoder)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(111)
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            student.load_state_dict(checkpoint['student'])
            teacher.load_state_dict(checkpoint['teacher'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_vit.load_state_dict(checkpoint['optimizer_vit'])
            scaler.load_state_dict(checkpoint['scaler'])
            log.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            for _ in range(3):
                print("=> no checkpoint found at '{}', train from scratch !!!!!!!!!!!!".format(args.resume))

    if args.fixTo > 0:
        fixBackBone(model, args.fixTo)

    cudnn.benchmark = True

    # Data loading code
    train_transform = pretrain_transform(args.crop_min, local_crops_number=args.local_crops_number)

    train_dataset, val_dataset, _ = init_datasets(args, train_transform, train_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)

    # ============ init schedulers ... ============
    lr_schedule = self_utils.cosine_scheduler(
        args.lr * (args.batch_size * self_utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = self_utils.cosine_scheduler(
        args.weight_decay2,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = self_utils.cosine_scheduler(args.momentum_teacher, 1,
                                                    args.epochs, len(train_loader))

    if args.evaluate_pretrain:
        from utils.pretrain import evaluate_pretrain, evaluate_pretrain_simRank
        evaluate_pretrain(train_loader, model, args, log)
        return

    if args.evaluate_VIC:
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

        evaluate_VIC(val_loader, model, args, log)
        return

    # print("开始训练")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(student, teacher, teacher_without_ddp,  teachermid_loss,  train_loader, model, optimizer_vit, lr_schedule,
              wd_schedule, momentum_schedule, optimizer, scaler, epoch, args, log, l1_regularizer=l1_regularizer)

        save_state_dict = model.state_dict()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):  # only the first GPU saves checkpoint

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_state_dict,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_vit': optimizer_vit.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename='checkpoint_%04d.pth.tar' % (epoch + 1),
                    save_dir=log.path)
            if not args.no_save_last:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_state_dict,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_vit': optimizer_vit.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, filename='checkpoint_last.pth.tar',
                    save_dir=log.path)

        if args.multiprocessing_distributed:
            torch.distributed.barrier()

    if args.start_epoch < args.epochs:
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank == 0):  # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': save_state_dict,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_vit': optimizer_vit.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename='checkpoint_final.pth.tar',
                save_dir=log.path)

        if args.multiprocessing_distributed:
            torch.distributed.barrier()


def train(student, teacher, teacher_without_ddp,   teachermid_loss,  train_loader, model, optimizer_vit, lr_schedule,
          wd_schedule, momentum_schedule, optimizer, scaler, epoch, args, log, l1_regularizer=None):
    for param_group in optimizer.param_groups:
        if "name" in param_group:
            log.info("group {}, current lr is {}".format(param_group["name"], param_group["lr"]))
        else:
            log.info("current lr is {}".format(param_group["lr"]))

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('self_ts_Loss', ':.4e')
    ce_losses = AverageMeter('CE_Loss', ':.4e')
    meters = [batch_time, data_time, learning_rates, losses, ce_losses]

    if args.l1_dist_w > 0:
        l1_dist_losses = AverageMeter('l1_dist', ':.4e')
        meters += [l1_dist_losses]
    else:
        meters = [batch_time, data_time, learning_rates, losses, ce_losses]

    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Epoch: [{}]".format(epoch),
        log=log)

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, (sample, targets) in enumerate(train_loader):
        images = sample
        bboxs = None
        it = len(train_loader) * epoch + i  # global training iteration 应该是这里导致一直 Epoch: [236][94480/400]
        # print("111111111")
        # for i, param_group in enumerate(optimizer_vit.param_groups):
        #     param_group["lr"] = lr_schedule[it]
        #     if i == 0:  # only the first group is regularized
        #         param_group["weight_decay"] = wd_schedule[it]
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        assert args.local_rank is not None

        images[0] = images[0].cuda(args.local_rank, non_blocking=True)
        images[1] = images[1].cuda(args.local_rank, non_blocking=True)

        x3 = None

        if args.local_crops_number > 0:
            x_local = [img.cuda(args.local_rank, non_blocking=True) for img in images[2]]
        else:
            x_local = None

        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        with torch.cuda.amp.autocast(True):
            assert x_local is None
            # move images to gpu
            # images = [im.cuda(non_blocking=True) for im in images]
            # images = images.to(device)

            images = [torch.squeeze(img, dim=1) for img in images]
            images = [im.cuda(non_blocking=True) for im in images]

            contrastive_loss = model(images[0], images[1],targets, moco_m, epoch=epoch)
            teacher_mid_loss = teachermid_loss(images[0], images[1])
            loss =0.5* contrastive_loss + 0.5 * teacher_mid_loss
        # print("2222")
        losses.update(0.00005 * teacher_mid_loss.item(), images[0].size(0))
        ce_losses.update(contrastive_loss.item(), images[0].size(0))
        if args.l1_dist_w > 0:
            # record the init weight
            loss_l1_regu = l1_regularizer(model.module.base_encoder, args.l1_dist_to_block)
            l1_dist_losses.update(loss_l1_regu.item(), images[0].size(0))
            loss = loss + args.l1_dist_w * loss_l1_regu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_vit.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                # logging

        if i % args.print_freq == 0:
            progress.display(i)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log = log

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.log.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()


