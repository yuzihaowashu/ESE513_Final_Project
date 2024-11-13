from unet_eval import unet_eval
from bm3d_eval import bm3d_eval
from tv_eval import tv_eval
import zipfile
import os
from os import listdir
from os.path import isfile, join
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--working_dir', type=str, required=True, help='working directory')
    parser.add_argument('--dataset', type=str, required=True, help='dataset dir or .zip file')
    parser.add_argument('--denoiser_type', type=str, required=True, help='Denoiser Type')
    parser.add_argument('--denoiser_path', type=str, help='Path to the model')
    parser.add_argument('--kernel_size', type=int, required=True, help='Path to the model')
    parser.add_argument('--num_pictures', type=int, default=0, help='Number of pictures evaluted')
    parser.add_argument('--num_iter', type=int, required=True, help='Number of Iterations')
    args = parser.parse_args()

    # Change directory
    os.chdir(args.working_dir)


    # Choose denoiser
    if(args.denoiser_type == 'unet'):
        unet_eval(args.dataset, args.denoiser_path, args.kernel_size, args.num_pictures, args.num_iter)
    elif(args.denoiser_type == 'bm3d'):
        bm3d_eval(args.dataset, args.kernel_size, args.num_pictures, args.num_iter)
    elif(args.denoiser_type == 'tv'):
        tv_eval(args.dataset, args.kernel_size, args.num_pictures, args.num_iter)

