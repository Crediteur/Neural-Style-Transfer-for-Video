import torch
import torchvision

# test if gpu is detected and check torch versions
print(
    f"device: {"cuda" if torch.cuda.is_available() else "cpu"}, num: {torch.cuda.current_device()}"
)
# device: cuda, num: 0

print(f"Pytorch version: {torch.__version__}")
# 2.8.0+cu126

print(f"Pytorch vision version: {torchvision.__version__}")
# 0.23.0+cu126
