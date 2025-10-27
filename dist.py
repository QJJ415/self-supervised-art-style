import os
import torch


def dist_init(port=23456):
    # 初始化分布式训练环境
    def init_parrots(host_addr, rank, local_rank, world_size, port):
        # 设置环境变量，用于指定主节点的地址和端口
        os.environ['MASTER_ADDR'] = str(host_addr)
        os.environ['MASTER_PORT'] = str(port)
        # 设置环境变量，指定分布式训练的总进程数和当前进程的排名
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        # 使用 NCCL 作为分布式训练的后端初始化进程组
        torch.distributed.init_process_group(backend="nccl")
        # 设置当前 GPU 设备为本地进程的 GPU 设备
        torch.cuda.set_device(local_rank)

    def init(host_addr, rank, local_rank, world_size, port):
        # 构造主节点的全地址，使用 "tcp://" 协议
        host_addr_full = 'tcp://' + host_addr + ':' + str(port)
        # 使用 PyTorch 提供的 init_process_group 函数初始化分布式训练的进程组
        # 使用 "gloo" 作为后端，指定主节点地址、排名和总进程数
        torch.distributed.init_process_group("gloo", init_method=host_addr_full,
                                             rank=rank, world_size=world_size)
        # 设置当前 GPU 设备为本地进程的 GPU 设备
        torch.cuda.set_device(local_rank)
        # 断言当前进程已经成功初始化分布式训练环境
        assert torch.distributed.is_initialized()

    def parse_host_addr(s):  # 解析字符串s，该字符串表示一个主机地址
        # 检查字符串是否包含 '['
        if '[' in s:
            # 找到 '[' 和 ']' 的索引位置
            left_bracket = s.index('[')
            right_bracket = s.index(']')
            # 提取 '[' 之前的前缀
            prefix = s[:left_bracket]
            # 从 '[' 和 ']' 之间提取第一个数字
            first_number = s[left_bracket + 1:right_bracket].split(',')[0].split('-')[0]
            # 将前缀和第一个数字合并
            return prefix + first_number
        else:
            return s

    rank = 0
    local_rank = 0
    world_size = 1
    ip = 'localhost'  # 设置为适当的IP地址或主机名

    if 'SLURM_PROCID' in os.environ:   # os.environ 是一个字典，是环境变量的字典
        rank = int(os.environ['SLURM_PROCID'])
    if 'SLURM_LOCALID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
    if 'SLURM_NTASKS' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
    if 'SLURM_STEP_NODELIST' in os.environ:
        ip = parse_host_addr(os.environ['SLURM_STEP_NODELIST'])

    if torch.__version__ == 'parrots':
        init_parrots(ip, rank, local_rank, world_size, port)
    else:
        init(ip, rank, local_rank, world_size, port)

    return rank, local_rank, world_size

