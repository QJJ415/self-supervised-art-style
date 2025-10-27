RUN

python -u main_pretrain.py --dataset AVA  --m 0.99

TEST

python -u main_finetune_mae.py --dataset AVA --checkpoint bgt-AVA.pth --s cos
