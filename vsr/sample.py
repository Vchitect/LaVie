import io
import os
import sys
import argparse
o_path = os.getcwd()
sys.path.append(o_path)

import torch
import time
import json
import numpy as np
import imageio
import torchvision
from einops import rearrange

from models.autoencoder_kl import AutoencoderKL
from models.unet import UNet3DVSRModel
from models.pipeline_stable_diffusion_upscale_video_3d import StableDiffusionUpscalePipeline
from diffusers import DDIMScheduler
from omegaconf import OmegaConf


def main(args):

	device = "cuda" 

	# ---------------------- load models ----------------------
	pipeline = StableDiffusionUpscalePipeline.from_pretrained(args.pretrained_path + '/stable-diffusion-x4-upscaler', torch_dtype=torch.float16)

	# vae
	pipeline.vae = AutoencoderKL.from_config("configs/vae_config.json")
	pretrained_model = args.pretrained_path + "/stable-diffusion-x4-upscaler/vae/diffusion_pytorch_model.bin"
	pipeline.vae.load_state_dict(torch.load(pretrained_model, map_location="cpu"))

	# unet
	config_path = "./configs/unet_3d_config.json"
	with open(config_path, "r") as f:
		config = json.load(f)
	config['video_condition'] = False
	pipeline.unet = UNet3DVSRModel.from_config(config)

	pretrained_model = args.ckpt_path
	checkpoint = torch.load(pretrained_model, map_location="cpu")['ema']

	pipeline.unet.load_state_dict(checkpoint, True) 
	pipeline.unet = pipeline.unet.half()
	pipeline.unet.eval() # important!

	# DDIMScheduler
	with open(args.pretrained_path + '/stable-diffusion-x4-upscaler/scheduler/scheduler_config.json', "r") as f:
		config = json.load(f)
	config["beta_schedule"] = "linear"
	pipeline.scheduler = DDIMScheduler.from_config(config)

	pipeline = pipeline.to("cuda")

	# ---------------------- load user's prompt ----------------------
	# input
	video_root = args.input_path
	video_list = sorted(os.listdir(video_root))
	print('video num:', len(video_list))

	# output
	save_root = args.output_path
	os.makedirs(save_root, exist_ok=True)

	# inference params
	noise_level = args.noise_level
	guidance_scale = args.guidance_scale
	num_inference_steps = args.inference_steps

	# ---------------------- start inferencing ----------------------
	for i, video_name in enumerate(video_list):
		video_name = video_name.replace('.mp4', '')			   
		print(f'[{i+1}/{len(video_list)}]: ', video_name)
		
		lr_path = f"{video_root}/{video_name}.mp4"
		save_path = f"{save_root}/{video_name}.mp4"

		prompt = video_name
		print('Prompt: ', prompt)

		negative_prompt = "blur, worst quality"

		vframes, aframes, info = torchvision.io.read_video(filename=lr_path, pts_unit='sec', output_format='TCHW') # RGB
		vframes = vframes / 255.
		vframes = (vframes - 0.5) * 2 # T C H W [-1, 1]
		t, _, h, w = vframes.shape
		vframes = vframes.unsqueeze(dim=0) # 1 T C H W
		vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()  # 1 C T H W
		print('Input_shape:', vframes.shape, 'Noise_level:', noise_level, 'Guidance_scale:', guidance_scale)

		fps = info['video_fps']
		generator = torch.Generator(device=device).manual_seed(10)

		torch.cuda.synchronize()
		start_time = time.time()

		with torch.no_grad():
			short_seq = 8
			vframes_seq = vframes.shape[2]
			if vframes_seq > short_seq: # for VSR
				upscaled_video_list = []
				for start_f in range(0, vframes_seq, short_seq):
					print(f'Processing: [{start_f}-{start_f + short_seq}/{vframes_seq}]')
					torch.cuda.empty_cache() # delete for VSR
					end_f = min(vframes_seq, start_f + short_seq)
					
					upscaled_video_ = pipeline(
						prompt,
						image=vframes[:,:,start_f:end_f],
						generator=generator,
						num_inference_steps=num_inference_steps,
						guidance_scale=guidance_scale,
						noise_level=noise_level,
						negative_prompt=negative_prompt,
					).images # T C H W [-1, 1]
					upscaled_video_list.append(upscaled_video_)
				upscaled_video = torch.cat(upscaled_video_list, dim=0)
			else:
				upscaled_video = pipeline(
					prompt,
					image=vframes,
					generator=generator,
					num_inference_steps=num_inference_steps,
					guidance_scale=guidance_scale,
					noise_level=noise_level,
					negative_prompt=negative_prompt,
				).images # T C H W [-1, 1]

		torch.cuda.synchronize()
		run_time = time.time() - start_time

		print('Output:', upscaled_video.shape)
		
		# save video
		upscaled_video = (upscaled_video / 2 + 0.5).clamp(0, 1) * 255
		upscaled_video = upscaled_video.permute(0, 2, 3, 1).to(torch.uint8)
		upscaled_video = upscaled_video.numpy().astype(np.uint8)
		imageio.mimwrite(save_path, upscaled_video, fps=fps, quality=9) # Highest quality is 10, lowest is 0

		print(f'Save upscaled video "{video_name}" in {save_path}, time (sec): {run_time} \n')
	print(f'\nAll results are saved in {save_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))
