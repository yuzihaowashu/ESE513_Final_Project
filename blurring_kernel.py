
import torch
import torch.nn as nn
import torch.nn.functional as F

# # How to use:
# from blurring_kernel import get_blur_operator

# # Parameters
# kernel_size = 21
# blur_type = 'gaussian'  # Options: 'gaussian', 'box', 'motion'
# channels = 3  # For RGB images
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# forward, forward_adjoint = get_blur_operator(
#     blur_type, channels, device, kernel_size, sigma=5, angle=0
# )

# y = forward(test_image)

# with torch.no_grad():
#     model.eval()
#     x = pnp_admm_cg(y, forward, forward_adjoint, model)
#     x = x.clip(0, 1)

# # Plot results
# print('PSNR [dB]: {:.2f}'.format(compute_psnr(x, test_image)))
# myplot(y,x,test_image)

def get_gaussian_kernel(kernel_size, sigma):
    """Generates a 2D Gaussian kernel."""
    ax = torch.arange(-((kernel_size - 1) // 2), ((kernel_size - 1) // 2) + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid([ax, ax], indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def get_box_kernel(kernel_size):
    """Generates a 2D Box kernel."""
    kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
    kernel = kernel / kernel.sum()
    return kernel

def get_motion_blur_kernel(kernel_size, angle):
    """Generates a 2D Motion blur kernel at a specified angle."""
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    if angle == 0:
        kernel[kernel_size // 2, :] = torch.ones(kernel_size)
    elif angle == 90:
        kernel[:, kernel_size // 2] = torch.ones(kernel_size)
    else:
        # For arbitrary angles, rotate the kernel
        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        coords = torch.stack(torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size)), -1).float()
        center = (kernel_size - 1) / 2.
        coords = coords - center
        rotated_coords = torch.matmul(coords, torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]]))
        rotated_coords = rotated_coords + center
        rotated_coords = rotated_coords.round().long()
        mask = ((rotated_coords[..., 0] >= 0) & (rotated_coords[..., 0] < kernel_size) &
                (rotated_coords[..., 1] >= 0) & (rotated_coords[..., 1] < kernel_size))
        kernel[coords[..., 0][mask], coords[..., 1][mask]] = 1
    kernel = kernel / kernel.sum()
    return kernel

def get_blur_kernel(blur_type, kernel_size, **kwargs):
    """Returns the kernel based on the blur type."""
    if blur_type == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return get_gaussian_kernel(kernel_size, sigma)
    elif blur_type == 'box':
        return get_box_kernel(kernel_size)
    elif blur_type == 'motion':
        angle = kwargs.get('angle', 0)
        return get_motion_blur_kernel(kernel_size, angle)
    else:
        raise ValueError('Unknown blur type')

def conv2d_from_kernel(kernel, channels, device, stride=1):
    """
    Returns nn.Conv2d and nn.ConvTranspose2d modules from 2D kernel,
    ensuring that the output size matches the input size.
    """
    kernel_size = kernel.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = (kernel_size[0] - 1) // 2  # Ensures output size equals input size when stride=1

    filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=padding
    )
    filter.weight.data = kernel
    filter.weight.requires_grad = False

    filter_adjoint = nn.ConvTranspose2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=padding
    )
    # Flip the kernel for the adjoint operator
    filter_adjoint.weight.data = kernel.flip(-2, -1)
    filter_adjoint.weight.requires_grad = False

    return filter.to(device), filter_adjoint.to(device)

def get_blur_operator(blur_type, channels, device, kernel_size, **kwargs):
    """
    Returns the forward and adjoint operators for the specified blur type.
    """
    if blur_type in ['gaussian', 'box', 'motion']:
        kernel = get_blur_kernel(blur_type, kernel_size, **kwargs)
        forward, forward_adjoint = conv2d_from_kernel(kernel, channels, device)
    else:
        raise ValueError('Unknown blur type')
    return forward, forward_adjoint