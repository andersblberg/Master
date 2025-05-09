import torch, torchvision

print(torch.__version__)  # e.g. 2.1.0+cu118
print(torch.cuda.is_available())  # → True
print(torch.cuda.get_device_name())  # → NVIDIA GeForce RTX 3070 Laptop GPU
