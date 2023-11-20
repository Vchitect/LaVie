from .unet import UNet3DVSRModel
from torch.optim.lr_scheduler import LambdaLR

def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models():
    config_path = "./configs/unet_3d_config.json"
    pretrained_model_path = "./pretrained_models/upscaler4x/unet/diffusion_pytorch_model.bin"
    return UNet3DVSRModel.from_pretrained_2d(config_path, pretrained_model_path)

    