pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# RTX 5090 D (sm_120) 必须用 CUDA 12.8 + torch cu128，镜像里已预装：
#   10.121.177.20:8082/mlops/ubuntu-gpu:cuda12.8.1-cudnn-python3.11
# 不要再 pip 装 torch==2.0.1，会覆盖镜像里的 cu128 包，再次出现 no kernel image。
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print('arch', torch.cuda.get_arch_list())
PY

pip install tensorboardX

#export NCCL_IB_HCA=mlx5   # 需要适配,ib或者roce 都行
#export NCCL_IB_TC=136
# export NCCL_IB_SL=5   # 需要适配，可不填
# export NCCL_IB_GID_INDEX=0    # 需要适配，可不填
#export NCCL_IB_TIMEOUT=22
# export NCCL_SOCKET_IFNAME=eth0  # 可不填，默认就是这个，有些协议无法走ib，会自动走以太网
#export NCCL_DEBUG=INFO

mkdir -p  data/MNIST/raw/
if [ "$RANK" = "0" ]; then

  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/train-images-idx3-ubyte.gz
  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/train-labels-idx1-ubyte.gz
  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/t10k-images-idx3-ubyte.gz
  wget -P  data/MNIST/raw/ https://docker-76009.sz.gfp.tencent-cloud.com/kubeflow/pytorch/example/data/t10k-labels-idx1-ubyte.gz
fi
python demo.py

