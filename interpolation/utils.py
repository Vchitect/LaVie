import os
import math
import torch
import logging
import subprocess
import numpy as np
import torch.distributed as dist

# from torch._six import inf
from torch import inf
from PIL import Image
from typing import Union, Iterable
from collections import OrderedDict
  

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

#################################################################################
#                             Training Clip Gradients                           #
#################################################################################

def get_grad_norm(
        parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm

def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    # print(total_norm)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        # print(gradient_cliped)
    return total_norm

#################################################################################
#                             Training Logger                                   #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def create_accelerate_logger(logging_dir, is_main_process=False):
    """
    Create a logger that writes to a log file and stdout.
    """
    if is_main_process:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:  # real tensorboard 
        # tensorboard 
        writer = SummaryWriter(tensorboard_dir)

    return writer

def write_tensorboard(writer, *args):
    '''
    write the loss information to a tensorboard file.
    Only for pytorch DDP mode.
    '''
    if dist.get_rank() == 0:  # real tensorboard
        writer.add_scalar(args[0], args[1], args[2])

#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()
    

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    print(f'Hahahahahaha')
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29566 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(rank % num_gpus)

    print(f'before dist.init_process_group')

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    print(f'after dist.init_process_group')

#################################################################################
#                             Testing  Utils                                    #
#################################################################################

def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape
    
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype=torch.uint8)
    
    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    
    return video_grid


#################################################################################
#                             MMCV  Utils                                    #
#################################################################################


def collect_env():
    # Copyright (c) OpenMMLab. All rights reserved.
    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash
    """Collect the information of the running environments."""
    
    env_info = collect_base_env()
    env_info['MMClassification'] = get_git_hash()[:7]

    for name, val in env_info.items():
        print(f'{name}: {val}')
    
    print(torch.cuda.get_arch_list())
    print(torch.version.cuda)

#################################################################################
#                      Long video generation  Utils                             #
#################################################################################

def mask_generation(mask_type, shape, dtype, device):
    b, c, f, h, w = shape
    if mask_type.startswith('random'):
        num = float(mask_type.split('random')[-1])
        mask_f = torch.ones(1, 1, f, 1, 1, dtype=dtype, device=device)
        indices = torch.randperm(f, device=device)[:int(f*num)]
        mask_f[0, 0, indices, :, :] = 0
        mask = mask_f.expand(b, c, -1, h, w)
    elif mask_type.startswith('first'):
        num = int(mask_type.split('first')[-1])
        mask_f = torch.cat([torch.zeros(1, 1, num, 1, 1, dtype=dtype, device=device),
                           torch.ones(1, 1, f-num, 1, 1, dtype=dtype, device=device)], dim=2)
        mask = mask_f.expand(b, c, -1, h, w)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    return mask



def mask_generation_before(mask_type, shape, dtype, device):
    b, f, c, h, w = shape
    if mask_type.startswith('random'):
        num = float(mask_type.split('random')[-1])
        mask_f = torch.ones(1, f, 1, 1, 1, dtype=dtype, device=device)
        indices = torch.randperm(f, device=device)[:int(f*num)]
        mask_f[0, indices, :, :, :] = 0
        mask = mask_f.expand(b, -1, c, h, w)
    elif mask_type.startswith('first'):
        num = int(mask_type.split('first')[-1])
        mask_f = torch.cat([torch.zeros(1, num, 1, 1, 1, dtype=dtype, device=device),
                           torch.ones(1, f-num, 1, 1, 1, dtype=dtype, device=device)], dim=1)
        mask = mask_f.expand(b, -1, c, h, w)
    elif mask_type.startswith('uniform'):
        p = float(mask_type.split('uniform')[-1])
        mask_f = torch.ones(1, f, 1, 1, 1, dtype=dtype, device=device)
        mask_f[0, torch.rand(f, device=device) < p, :, :, :] = 0
        print(f'mask_f: = {mask_f}')
        mask = mask_f.expand(b, -1, c, h, w)
        print(f'mask.shape: = {mask.shape}, mask: = {mask}')
    elif mask_type.startswith('all'):
        mask = torch.ones(b,f,c,h,w,dtype=dtype,device=device)
    elif mask_type.startswith('onelast'):
        num = int(mask_type.split('onelast')[-1])
        mask_one = torch.zeros(1,1,1,1,1, dtype=dtype, device=device)
        mask_mid = torch.ones(1,f-2*num,1,1,1,dtype=dtype, device=device)
        mask_last = torch.zeros_like(mask_one)
        mask = torch.cat([mask_one]*num + [mask_mid] + [mask_last]*num, dim=1)
        # breakpoint()
        mask = mask.expand(b, -1, c, h, w)
    elif mask_type.startswith('interpolate'):
        mask_f = []
        for i in range(4):
            mask_zero = torch.zeros(1,1,1,1,1, dtype=dtype, device=device)
            mask_f.append(mask_zero)
            mask_one = torch.ones(1,3,1,1,1, dtype=dtype, device=device)
            mask_f.append(mask_one)
        mask = torch.cat(mask_f, dim=1)
        print(f'mask={mask}')
    elif mask_type.startswith('tsr'):
        mask_f = []
        mask_zero = torch.zeros(1,1,1,1,1, dtype=dtype, device=device)
        mask_one = torch.ones(1,3,1,1,1, dtype=dtype, device=device)
        for i in range(15):
            mask_f.append(mask_zero) # not masked
            mask_f.append(mask_one) # masked
        mask_f.append(mask_zero) # not masked
        mask = torch.cat(mask_f, dim=1)
        # print(f'before mask.shape = {mask.shape}, mask = {mask}') # [1, 61, 1, 1, 1]
        mask = mask.expand(b, -1, c, h, w)
        # print(f'after mask.shape = {mask.shape}, mask = {mask}') # [4, 61, 3, 256, 256]
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")

    return mask
