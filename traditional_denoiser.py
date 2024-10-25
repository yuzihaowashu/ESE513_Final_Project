import torch
import numpy as np
import bm3d
from skimage.restoration import denoise_tv_chambolle

def bm3d_denoiser(image_tensor, sigma=0.1):
    """
    BM3D denoiser for use in the PnP-ADMM framework.
    
    Args:
        image_tensor (torch.Tensor): The noisy image tensor (B, C, H, W) or (C, H, W).
        sigma (float): Noise standard deviation (controls the amount of denoising).
        
    Returns:
        torch.Tensor: The denoised image as a torch tensor.
    """
    # Check if the input has a batch dimension (4D tensor)
    if len(image_tensor.shape) == 4:  # [B, C, H, W]
        batch_size = image_tensor.shape[0]
        denoised_batch = []
        
        # Loop over each image in the batch and apply BM3D individually
        for i in range(batch_size):
            img = image_tensor[i]  # Get one image: [C, H, W]
            img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] for BM3D

            # Prepare an empty array for the denoised image
            denoised_np = np.zeros_like(img_np)

            # Apply BM3D denoising to each channel independently
            for c in range(img_np.shape[2]):  # Loop over channels
                denoised_np[:, :, c] = bm3d.bm3d(img_np[:, :, c], sigma)

            # Convert back to PyTorch tensor
            denoised_img = torch.tensor(denoised_np).permute(2, 0, 1).to(image_tensor.device)  # Back to [C, H, W]
            denoised_batch.append(denoised_img)

        # Stack the denoised images back into a batch
        return torch.stack(denoised_batch)

    elif len(image_tensor.shape) == 3:  # [C, H, W] case (single image)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] for BM3D

        # Prepare an empty array for the denoised image
        denoised_np = np.zeros_like(img_np)

        # Apply BM3D denoising to each channel independently
        for c in range(img_np.shape[2]):  # Loop over channels
            denoised_np[:, :, c] = bm3d.bm3d(img_np[:, :, c], sigma)

        # Convert back to PyTorch tensor
        return torch.tensor(denoised_np).permute(2, 0, 1).to(image_tensor.device)  # Back to [C, H, W]
    
# Total Variation (TV) denoising function
def tv_denoiser(image, weight=0.1):
    # Apply TV denoising and convert the result back to a tensor
    denoised = denoise_tv_chambolle(image.cpu().numpy(), weight=weight)
    return torch.tensor(denoised, device=image.device)