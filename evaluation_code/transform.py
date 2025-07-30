import torch

def normalize_and_scale_image(tensor_img):
    """
    Scales a tensor from [0, 255] to [0, 1] and then normalizes
    it to [-1, 1] using mean=0.5 and std=0.5.
    """
    # 1. Scale from [0, 255] to [0.0, 1.0]
    tensor_img = tensor_img.div(255.0)

    # 2. Normalize from [0.0, 1.0] to [-1.0, 1.0]
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor_img.device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor_img.device).view(3, 1, 1)
    tensor_img = tensor_img.sub(mean).div(std)

    return tensor_img