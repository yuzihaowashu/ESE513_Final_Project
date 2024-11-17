from unet_eval import unet_eval
from bm3d_eval import bm3d_eval
from tv_eval import tv_eval
from fastdvdnet_eval import fastdvdnet_eval
from dpir_eval import dpir_eval
import zipfile
import os
from os import listdir
from os.path import isfile, join
import argparse
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--working_dir', type=str, required=True, help='working directory')
    parser.add_argument('--dataset', type=str, required=True, help='dataset dir or .zip file')
    parser.add_argument('--pnp_type', type=str, required=True, help='pnp_admm_least_square / pnp_admm_cg')
    parser.add_argument('--denoiser_type', type=str, required=True, help='Denoiser Type')
    parser.add_argument('--denoiser_path', type=str, help='Path to the model')
    parser.add_argument('--kernel_size', type=int, required=True, help='Path to the model')
    parser.add_argument('--num_pictures', type=int, default=0, help='Number of pictures evaluted')
    parser.add_argument('--num_iter', type=int, required=True, help='Number of Iterations')
    parser.add_argument('--step_size', type=float, default=1e-2, help='Step size for least square method')
    parser.add_argument('--max_cgiter', type=int, default=50, help='max cg iter')
    parser.add_argument('--cg_tol', type=float, default=1e-7, help='cg tolerence')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level')
    
    args = parser.parse_args()

    # Change directory
    os.chdir(args.working_dir)


    # Choose denoiser
    if(args.denoiser_type == 'unet'):
        unet_eval(args.dataset, args.pnp_type, args.denoiser_path, args.kernel_size, args.num_pictures, args.num_iter, args.step_size, args.max_cgiter, args.cg_tol)
    elif(args.denoiser_type == 'dpir'):
        dpir_eval(args.dataset, args.pnp_type, args.denoiser_path, args.kernel_size, args.num_pictures, args.num_iter, args.step_size, args.max_cgiter, args.cg_tol, args.noise_level)
    elif(args.denoiser_type == 'bm3d'):
        bm3d_eval(args.dataset, args.pnp_type, args.kernel_size, args.num_pictures, args.num_iter, args.step_size, args.max_cgiter, args.cg_tol)
    elif(args.denoiser_type == 'tv'):
        tv_eval(args.dataset, args.pnp_type, args.kernel_size, args.num_pictures, args.num_iter, args.step_size, args.max_cgiter, args.cg_tol) 
    elif(args.denoiser_type == 'fastdvdnet'): 
        fastdvdnet_eval(args.dataset, args.pnp_type, args.denoiser_path, args.kernel_size, args.num_pictures, args.num_iter, args.step_size, args.max_cgiter, args.cg_tol)

