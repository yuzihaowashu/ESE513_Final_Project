import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pnp import pnp_admm_fastdvdnet
from model import FastDVDnet
from utils import conv2d_from_kernel, compute_psnr, ImagenetDataset, myplot
from utils_fastdvdnet import batch_psnr, open_sequence
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
import time
import zipfile
from os import listdir
from os.path import isfile, join
import time
from torchvision import transforms
import glob
 

def fastdvdnet_eval(dataset, denoiser_path, kernel_size, num_pictures, num_iter):
    # mkdir
    if not os.path.exists('./eval_dataset/'):
        os.makedirs('./eval_dataset/')
    if not os.path.exists(f"./eval_output_fastdvdnet_{kernel_size}/"):
        os.makedirs(f"./eval_output_fastdvdnet_{kernel_size}/")
    # Load Data
    path = dataset
    if dataset.endswith('.zip'):
        with zipfile.ZipFile(dataset, 'r') as zip_ref:
            zip_ref.extractall('eval_dataset')
        path = './eval_dataset/'

    # Count number of frames
    def count_images(directory, extensions=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff")):
        image_count = 0
        for ext in extensions:
            image_count += len(glob.glob(os.path.join(directory, ext)))
        return image_count
    num_frames = count_images(path)



    # Define device and model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_temp = FastDVDnet(num_input_frames=5)
    state_temp_dict = torch.load(denoiser_path, map_location=device)
    device_ids = [0]
    model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    model_temp.load_state_dict(state_temp_dict)

    # Define motion blur kernel and other parameters
    kernel_motion_blur = torch.ones((1, kernel_size))
    forward, forward_adjoint = conv2d_from_kernel(kernel_motion_blur, 3, device)
    
    # Load images
    gray = False
    seq, _, _ = open_sequence(path, gray, expand_if_needed=False, max_num_fr=num_frames)
    seq = torch.from_numpy(seq).to(device)
    y_seq = forward(seq)

    start_time = time.time()

    with torch.no_grad():
        model_temp.eval()
        start_time = time.time()
        denoised_image = pnp_admm_fastdvdnet(y_seq, forward, forward_adjoint, model_temp, num_iter=num_iter, max_cgiter=5, cg_tol=1e-7)
        end_time = time.time()

    end_time = time.time() 
    total_time = end_time - start_time   

    # Save images
    to_pil_image = transforms.ToPILImage()
    for i in range(denoised_image.shape[0]):
        img = to_pil_image(denoised_image[i])
        img.save(f"./eval_output_fastdvdnet_{kernel_size}/image_{i}.jpg")

    # Calculate PSNR
    psnr = 0 #batch_psnr(denoised_image, seq, 1.)
    psnr_noisy = 0 #batch_psnr(y_seq.squeeze(), seq, 1.)

    file_path = f"eval_output_fastdvdnet_{kernel_size}.txt"

    with open(file_path, 'w') as file:
        file.write(f"The total time for eval is: {total_time}\nPSNR noisy {psnr_noisy:.4f}dB, PSNR result {psnr:.4f}dB")
