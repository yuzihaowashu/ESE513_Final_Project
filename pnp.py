import torch
from fastdvdnet import denoise_seq_fastdvdnet
from utils_dpir import utils_model
from utils_dpir import utils_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pnp_admm_least_square(
        measurements, forward, forward_adjoint, denoiser, 
        step_size=1e-4, num_iter=50
    ):
    """
    ADMM plug and play using direct least squares solving
    """
    x_h = forward_adjoint(measurements)

    def least_squares_solve(A, b):
        """
        Direct least squares solution using torch.linalg.solve
        Solves (A^T A + step_size * I) x = b
        """
        AtA_b = forward_adjoint(forward(b)) + step_size * b
        return AtA_b

    # Start
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    for _ in range(num_iter):
        # Solve the least squares subproblem
        b = x_h + step_size * (v - u)
        x = least_squares_solve(forward, b)
        
        # Denoising step
        v = denoiser(x + u)
        
        # Dual variable update
        u += (x - v)

    return v



def pnp_admm_cg(
        measurements, forward, forward_adjoint, denoiser, 
        step_size=1e-4, num_iter=50, max_cgiter=100, cg_tol=1e-7
    ):
    """
    ADMM plug and play
    """
    x_h =  forward_adjoint(measurements)

    def conjugate_gradient(A, b, x0, max_iter, tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A(x)
        d = r
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/torch.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/torch.norm(b) < tol:
                break
            beta = torch.sum(r**2)/rr
            d = r + beta*d        
        return x

    def cg_leftside(x):
        """
        Return left side of Ax=b, i.e., Ax
        """
        return forward_adjoint(forward(x)) + step_size*x

    def cg_rightside(x):
        """
        Returns right side of Ax=b, i.e. b
        """
        return x_h + step_size*x

    # Start
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    for _ in range(num_iter):
        b = cg_rightside(v-u)
        x = conjugate_gradient(cg_leftside, b, x, max_cgiter, cg_tol)
        v = denoiser(x+u)
        u += (x - v)
    return v


def pnp_admm_cg_record_images(
        measurements, forward, forward_adjoint, denoiser, 
        step_size=1e-4, num_iter=50, max_cgiter=100, cg_tol=1e-7
    ):
    """
    ADMM plug and play returns images denoised in each iteration, which are stored in an array. 
    """
    x_h =  forward_adjoint(measurements)

    def conjugate_gradient(A, b, x0, max_iter, tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A(x)
        d = r
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/torch.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/torch.norm(b) < tol:
                break
            beta = torch.sum(r**2)/rr
            d = r + beta*d        
        return x

    def cg_leftside(x):
        """
        Return left side of Ax=b, i.e., Ax
        """
        return forward_adjoint(forward(x)) + step_size*x

    def cg_rightside(x):
        """
        Returns right side of Ax=b, i.e. b
        """
        return x_h + step_size*x

    # Start
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    v_array = []
    for _ in range(num_iter):
        b = cg_rightside(v-u)
        x = conjugate_gradient(cg_leftside, b, x, max_cgiter, cg_tol)
        v = denoiser(x+u)
        v_array.append(v.clone())
        u += (x - v)
    return v_array


def pnp_admm_least_square_fastdvdnet(
        measurements, forward, forward_adjoint, model_temporal, 
        step_size=1e-4, num_iter=50
    ):
    """
    ADMM plug and play using direct least squares solving
    """
    x_h = forward_adjoint(measurements)

    def least_squares_solve(A, b):
        """
        Direct least squares solution using torch.linalg.solve
        Solves (A^T A + step_size * I) x = b
        """
        AtA_b = forward_adjoint(forward(b)) + step_size * b
        return AtA_b

    # Start
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    for _ in range(num_iter):
        # Solve the least squares subproblem
        b = x_h + step_size * (v - u)
        x = least_squares_solve(forward, b)
        
        # Denoising step
        v = denoise_seq_fastdvdnet(x+u, model_temporal)
        
        # Dual variable update
        u += (x - v)

    return v


def pnp_admm_cg_fastdvdnet(
        measurements, forward, forward_adjoint, model_temporal, 
        step_size=1e-4, num_iter=50, max_cgiter=100, cg_tol=1e-7
    ):
    """
    ADMM plug and play
    """
    x_h =  forward_adjoint(measurements)

    def conjugate_gradient(A, b, x0, max_iter, tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A(x)
        d = r
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/torch.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/torch.norm(b) < tol:
                break
            beta = torch.sum(r**2)/rr
            d = r + beta*d        
        return x

    def cg_leftside(x):
        """
        Return left side of Ax=b, i.e., Ax
        """
        return forward_adjoint(forward(x)) + step_size*x

    def cg_rightside(x):
        """
        Returns right side of Ax=b, i.e. b
        """
        return x_h + step_size*x


    # Start
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    for _ in range(num_iter):
        b = cg_rightside(v-u)
        x = conjugate_gradient(cg_leftside, b, x, max_cgiter, cg_tol)
        v = denoise_seq_fastdvdnet(x+u, model_temporal)
        u += (x - v)
    return v

# Add on Nov.17th
# DPIR takes an additional channel of noise 
# We add the blur effect first and then add the noise channel
def pnp_admm_cg_dpir(
        measurements, forward, forward_adjoint, denoiser, 
        step_size=1e-4, num_iter=50, max_cgiter=100, cg_tol=1e-7, noise_level=1, 
    ):
    
    x_h =  forward_adjoint(measurements)

    def conjugate_gradient(A, b, x0, max_iter, tol):
        """
        Conjugate gradient method for solving Ax=b
        """
        x = x0
        r = b-A(x)
        d = r
        for _ in range(max_iter):
            z = A(d)
            rr = torch.sum(r**2)
            alpha = rr/torch.sum(d*z)
            x += alpha*d
            r -= alpha*z
            if torch.norm(r)/torch.norm(b) < tol:
                break
            beta = torch.sum(r**2)/rr
            d = r + beta*d        
        return x

    def cg_leftside(x):
        """
        Return left side of Ax=b, i.e., Ax
        """
        return forward_adjoint(forward(x)) + step_size*x

    def cg_rightside(x):
        """
        Returns right side of Ax=b, i.e. b
        """
        return x_h + step_size*x

    # Start
    x8 = False
    x = torch.zeros_like(x_h)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    for _ in range(num_iter):
        b = cg_rightside(v-u)
        x = conjugate_gradient(cg_leftside, b, x, max_cgiter, cg_tol)

        # Add an noise channel
        x_plus_u = x+u
        x_plus_u = torch.cat((
            x_plus_u, 
            torch.FloatTensor([noise_level / 255.]).to(device).repeat(1, 1, x_plus_u.shape[2], x_plus_u.shape[3])
        ), dim=1)

        
        if not x8 and x_plus_u.size(2)//8==0 and x_plus_u.size(3)//8==0:
            v = denoiser(x_plus_u)
        elif not x8 and (x_plus_u.size(2)//8!=0 or x_plus_u.size(3)//8!=0):
            v = utils_model.test_mode(denoiser, x_plus_u, refield=64, mode=5)
        elif x8:
            v = utils_model.test_mode(denoiser, x_plus_u, mode=3)
        u += (x - v)

    return torch.from_numpy(utils_image.tensor2uint(u))