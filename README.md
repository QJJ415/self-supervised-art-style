RUN

python -u main_pretrain.py --dataset AVA  --epochs 300

TEST

python -u main_finetune_mae.py --dataset AVA --checkpoint bgt-AVA.pth --s cos
