import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pnp import pnp_admm_cg, pnp_admm_least_square
from model import Unet, Unet_attention
from utils import conv2d_from_kernel, compute_psnr, ImagenetDataset, myplot
import PIL.Image as Image
import torch
import torch.utils.benchmark as benchmark
import time
import numpy as np
import torch.nn.functional as F
import zipfile
from os import listdir
from os.path import isfile, join
from blurring_kernel import get_blur_operator


def unet_eval(dataset, pnp_type, denoiser_path, kernel_size, num_pictures, num_iter, step_size, max_cgiter, cg_tol, blur_type, angle, blur_kernel_size):
    # mkdir
    if not os.path.exists('./eval_dataset/'):
        os.makedirs('./eval_dataset/')
    if not os.path.exists(f"./eval_output_unet_{kernel_size}/"):
        os.makedirs(f"./eval_output_unet_{kernel_size}/")
    # Load Data
    if dataset.endswith('.zip'):
        with zipfile.ZipFile(dataset, 'r') as zip_ref:
            zip_ref.extractall('eval_dataset')
    path = './eval_dataset/'
    img_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('jpg')]


    # Define device and model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Unet(3, 3, chans=64).to(device)
    model.load_state_dict(torch.load(denoiser_path, map_location=device))
    model.eval()

    # Define motion blur kernel and other parameters
    # kernel_motion_blur = torch.ones((1, kernel_size))
    # forward, forward_adjoint = conv2d_from_kernel(kernel_motion_blur, 3, device)

    channels = 3 # for RGB
    forward, forward_adjoint = get_blur_operator(
        blur_type, channels, device, blur_kernel_size, sigma=5, angle=angle
    )

    psnr = []
    i = 0
    # Process each image
    start_time = time.time()
    for img_file in img_files:
        i += 1
        if (i == num_pictures): break
        # Load and preprocess image
        test_image = Image.open(img_file).convert("RGB")
        test_image = ImagenetDataset([]).test_transform(test_image)
        test_image = test_image.unsqueeze(0).to(device)
    
        # Apply forward operator (e.g., motion blur)
        y = forward(test_image)
        
        # Denoise with U-Net model using ADMM
        with torch.no_grad():

            if pnp_type == 'pnp_admm_least_square': 
                denoised_image = pnp_admm_least_square(y, forward, forward_adjoint, model, step_size=step_size, num_iter=num_iter)
            else:
                denoised_image = pnp_admm_cg(y, forward, forward_adjoint, model, num_iter=num_iter, max_cgiter=5, cg_tol=1e-7)
            denoised_image = denoised_image.clip(0, 1)  # Clip to valid range

            # Calculate PSNR
            psnr_value = compute_psnr(denoised_image, test_image)
            psnr.append(psnr_value)

            # Save the result
            output_image = denoised_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_path = f"./eval_output_unet_{kernel_size}/{img_file.split('/')[-1].replace('.jpg', '_denoised.jpg')}"
            Image.fromarray((output_image * 255).astype('uint8')).save(output_path)
    
    end_time = time.time()      
    total_time = end_time - start_time
    psnr_average = torch.mean(torch.stack(psnr)).item()

    file_path = f"eval_output_unet_{kernel_size}.txt"

    with open(file_path, 'w') as file:
        file.write(f"The total time for eval is: {total_time}\nThe average PSNR is: {psnr_average}")