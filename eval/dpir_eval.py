import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dpir_models.network_unet import UNetRes as net
from utils_dpir import utils_image as util
import numpy as np
from pnp import pnp_admm_cg_dpir
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


def dpir_eval(dataset, pnp_type, denoiser_path, kernel_size, num_pictures, num_iter, step_size, max_cgiter, cg_tol, noise_level=0.01):
    # mkdir
    if not os.path.exists('./eval_dataset/'):
        os.makedirs('./eval_dataset/')
    if not os.path.exists(f"./eval_output_drunet_{kernel_size}/"):
        os.makedirs(f"./eval_output_drunet_{kernel_size}/")
    # Load Data
    if dataset.endswith('.zip'):
        with zipfile.ZipFile(dataset, 'r') as zip_ref:
            zip_ref.extractall('eval_dataset')
    path = './eval_dataset/'
    img_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith('jpg')]


    # Define device and model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_channels = 3
    model_path = denoiser_path
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # Define motion blur kernel and other parameters
    kernel_motion_blur = torch.ones((1, kernel_size))
    forward, forward_adjoint = conv2d_from_kernel(kernel_motion_blur, 3, device)

    psnr = []
    i = 0
    # Process each image
    start_time = time.time()
    for img_file in img_files:
        i += 1
        if (i == num_pictures): break

        img_H = util.imread_uint(img_file, n_channels=n_channels)
        img_L = util.uint2single(img_H)
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level / 255., img_L.shape)
        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)
    
        # Apply forward operator (e.g., motion blur)
        y = forward(img_L)
        
        # Denoise with U-Net model using ADMM
        with torch.no_grad():

            denoised_image = pnp_admm_cg_dpir(y, forward, forward_adjoint, model, step_size=step_size, num_iter=num_iter, max_cgiter=max_cgiter, cg_tol=cg_tol)
            #denoised_image = denoised_image.clip(0, 1)  # Clip to valid range

            # Calculate PSNR
            psnr_value = 0 #compute_psnr(denoised_image, img_L)
            psnr.append(psnr_value)

            # Save the result
            image_numpy = denoised_image.cpu().numpy()
            image_numpy = (image_numpy * 255).astype('uint8')
            output_path = f"./eval_output_drunet_{kernel_size}/{img_file.split('/')[-1].replace('.jpg', '_denoised.jpg')}"
            Image.fromarray(image_numpy).save(output_path)
    
    end_time = time.time()      
    total_time = end_time - start_time
    psnr_average = 0

    file_path = f"eval_output_drunet_{kernel_size}.txt"

    with open(file_path, 'w') as file:
        file.write(f"The total time for eval is: {total_time}\nThe average PSNR is: {psnr_average}")