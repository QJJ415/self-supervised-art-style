Applied Soft Computing Journal Self-Supervised Art Style Classification via Attention, Distillation, and Learnable Style
=================================================
Introduction
-----------------------
This is a PyTorch implementation.


Art style classification is a fundamental task in computational aesthetics, yet most existing methods rely heavily on supervised learning, which requires costly manual annotations. To address this issue, we propose a novel self-supervised learning
framework for art style classification that leverages unlabeled data to learn
discriminative and style-aware representations. Built upon contrastive learning, our
framework incorporates four key components: (1)  an attention-guided fusion module
that adaptively integrates multi-scale representations using self-attention;  (2) a multi—layer self-distillation mechanism, where the teacher’s final output guides the student’s
intermediate features;  (3) a similarity-aware contrastive learning strategy that replaces
rigid positive-negative sampling with soft semantic similarity weighting; and  (4) a
learnable Gram matrix module for dynamic modeling of channel-wise correlations to
enhance style encoding. Extensive experiments on three public datasets—AVA,
Pandora18k, and Flickr—demonstrate that our method outperforms state-of-the-art.

![Image text](https://github.com/QJJ415/self-supervised-art-style/blob/cc7aa8f6d2880b8e0d7365aa95cac9c77be1be8e/images.png)
Usage
-------------
## Pretrain

python -u main_pretrain.py --dataset AVA  --epochs 300

TEST

python -u main_finetune_mae.py --dataset AVA --checkpoint bgt-AVA.pth --epochs 100
