"""
we introduce a temporal interpolation network to enhance the smoothness of generated videos and synthesize richer temporal details.
This network takes a 16-frame base video as input and produces an upsampled output consisting of 61 frames.
"""

import os
import sys
import math
try:
	import utils

	from diffusion import create_diffusion
	from download import find_model
except:
	sys.path.append(os.path.split(sys.path[0])[0])
	
	import utils

	from diffusion import create_diffusion
	from download import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torchvision import transforms
sys.path.append("..")
from datasets import video_transforms
from decord import VideoReader
from utils import mask_generation, mask_generation_before
from natsort import natsorted
from diffusers.utils.import_utils import is_xformers_available

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_input(args):
	input_path = args.input_path
	transform_video = transforms.Compose([
			video_transforms.ToTensorVideo(), # TCHW
			video_transforms.ResizeVideo((args.image_h, args.image_w)),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
		])
	temporal_sample_func = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval)
	if input_path is not None:
		print(f'loading video from {input_path}')
		if os.path.isdir(input_path):
			file_list = os.listdir(input_path)
			video_frames = []
			for file in file_list:
				if file.endswith('jpg') or file.endswith('png'):
					image = torch.as_tensor(np.array(Image.open(file), dtype=np.uint8, copy=True)).unsqueeze(0)
					video_frames.append(image)
				else:
					continue
			n = 0
			video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
			video_frames = transform_video(video_frames)
			return video_frames, n
		elif os.path.isfile(input_path):
			_, full_file_name = os.path.split(input_path)
			file_name, extention = os.path.splitext(full_file_name)
			if extention == '.mp4':
				video_reader = VideoReader(input_path)
				total_frames = len(video_reader)
				start_frame_ind, end_frame_ind = temporal_sample_func(total_frames)
				frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, args.num_frames, dtype=int)
				video_frames = torch.from_numpy(video_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()
				video_frames = transform_video(video_frames)
				n = 3
				del video_reader
				return video_frames, n
			else:
				raise TypeError(f'{extention} is not supported !!')
		else:
			raise ValueError('Please check your path input!!')
	else:
		print('given video is None, using text to video')
		video_frames = torch.zeros(16,3,args.latent_h,args.latent_w,dtype=torch.uint8)
		args.mask_type = 'all'
		video_frames = transform_video(video_frames)
		n = 0
		return video_frames, n


def auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,):
	
	
	b,f,c,h,w=video_input.shape
	latent_h = args.image_size[0] // 8
	latent_w = args.image_size[1] // 8

	z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, device=device) # b,c,f,h,w

	masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
	masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
	masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
	mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)
	

	masked_video = torch.cat([masked_video] * 2) if args.do_classifier_free_guidance else masked_video
	mask = torch.cat([mask] * 2) if args.do_classifier_free_guidance else mask
	z = torch.cat([z] * 2) if args.do_classifier_free_guidance else z

	prompt_all = [prompt] + [args.negative_prompt] if args.do_classifier_free_guidance else [prompt]
	text_prompt = text_encoder(text_prompts=prompt_all, train=False)
	model_kwargs = dict(encoder_hidden_states=text_prompt, class_labels=None)

	if args.use_ddim_sample_loop:
		samples = diffusion.ddim_sample_loop(
			model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, \
			progress=True, device=device, mask=mask, x_start=masked_video, use_concat=args.use_concat
		)
	else:
		samples = diffusion.p_sample_loop(
			model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, \
				progress=True, device=device, mask=mask, x_start=masked_video, use_concat=args.use_concat
		) # torch.Size([2, 4, 16, 32, 32])
	samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]

	video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
	video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
	return video_clip


def auto_inpainting_copy_no_mask(args, video_input, prompt, vae, text_encoder, diffusion, model, device,):

	b,f,c,h,w=video_input.shape
	latent_h = args.image_size[0] // 8
	latent_w = args.image_size[1] // 8

	video_input = rearrange(video_input, 'b f c h w -> (b f) c h w').contiguous()
	video_input = vae.encode(video_input).latent_dist.sample().mul_(0.18215)
	video_input = rearrange(video_input, '(b f) c h w -> b c f h w', b=b).contiguous()

	lr_indice = torch.IntTensor([i for i in range(0,62,4)]).to(device)
	copied_video = torch.index_select(video_input, 2, lr_indice)
	copied_video = torch.repeat_interleave(copied_video, 4, dim=2)
	copied_video = copied_video[:,:,1:-2,:,:]
	copied_video = torch.cat([copied_video] * 2) if args.do_classifier_free_guidance else copied_video

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, device=device) # b,c,f,h,w
	z = torch.cat([z] * 2) if args.do_classifier_free_guidance else z
	
	prompt_all = [prompt] + [args.negative_prompt] if args.do_classifier_free_guidance else [prompt]
	text_prompt = text_encoder(text_prompts=prompt_all, train=False)
	model_kwargs = dict(encoder_hidden_states=text_prompt, class_labels=None)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	if args.use_ddim_sample_loop:
		samples = diffusion.ddim_sample_loop(
			model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, \
			progress=True, device=device, mask=None, x_start=copied_video, use_concat=args.use_concat, copy_no_mask=args.copy_no_mask,
		)
	else:
		raise ValueError(f'We only have ddim sampling implementation for now')

	samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]

	video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
	video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
	return video_clip



def main(args):

	for seed in args.seed_list:

		args.seed = seed
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		# print(f'torch.seed() = {torch.seed()}')

		print('sampling begins')
		torch.set_grad_enabled(False)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		# device = "cpu"
		
		ckpt_path = args.ckpt_path
		sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
		for ckpt in [ckpt_path]:

			ckpt_num = str(ckpt_path).zfill(7)

			# Load model:
			latent_h = args.image_size[0] // 8
			latent_w = args.image_size[1] // 8
			args.image_h = args.image_size[0]
			args.image_w = args.image_size[1]
			args.latent_h = latent_h
			args.latent_w = latent_w
			print(f'args.copy_no_mask = {args.copy_no_mask}')
			model = get_models(args, sd_path).to(device)

			if args.use_compile:
				model = torch.compile(model)
			if args.enable_xformers_memory_efficient_attention:
				if is_xformers_available():
					model.enable_xformers_memory_efficient_attention()
					# model.enable_vae_slicing() # ziqi added
				else:
					raise ValueError("xformers is not available. Make sure it is installed correctly")

			# Auto-download a pre-trained model or load a custom checkpoint from train.py:
			print(f'loading model from {ckpt_path}')
			
			# load ckpt
			state_dict = find_model(ckpt_path)
			print('loading succeed')
			model.load_state_dict(state_dict)

			torch.manual_seed(args.seed)
			torch.cuda.manual_seed(args.seed)

			model.eval()  # important!
			diffusion = create_diffusion(str(args.num_sampling_steps))
			vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(device)
			text_encoder = TextEmbedder(sd_path).to(device)

			video_list = os.listdir(args.input_folder)
			args.input_path_list = [os.path.join(args.input_folder, video) for video in video_list]
			for input_path in args.input_path_list:

				args.input_path = input_path

				print(f'=======================================')
				if not args.input_path.endswith('.mp4'):
					print(f'Skipping {args.input_path}')
					continue
					
				print(f'args.input_path = {args.input_path}')

				torch.manual_seed(args.seed)
				torch.cuda.manual_seed(args.seed)				 

				# Labels to condition the model with (feel free to change):
				video_name = args.input_path.split('/')[-1].split('.mp4')[0]
				args.prompt = [video_name]
				print(f'args.prompt = {args.prompt}')
				prompts = args.prompt
				class_name = [p + args.additional_prompt for p in prompts]

				if not os.path.exists(os.path.join(args.output_folder)):
					os.makedirs(os.path.join(args.output_folder))
				video_input, researve_frames = get_input(args) # f,c,h,w
				video_input = video_input.to(device).unsqueeze(0) # b,f,c,h,w
				if args.copy_no_mask:
					pass
				else:
					mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device) # b,f,c,h,w

				if args.copy_no_mask:
					pass
				else:
					if args.mask_type == 'tsr':
						masked_video = video_input * (mask == 0)
					else:
						masked_video = video_input * (mask == 0)

				all_video = []
				if researve_frames != 0:
					all_video.append(video_input)
				for idx, prompt in enumerate(class_name):
					if idx == 0:
						if args.copy_no_mask:
							video_clip = auto_inpainting_copy_no_mask(args, video_input, prompt, vae, text_encoder, diffusion, model, device,)
							video_clip_ = video_clip.unsqueeze(0)
							all_video.append(video_clip_)
						else:
							video_clip = auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,)
							video_clip_ = video_clip.unsqueeze(0)
							all_video.append(video_clip_)
					else:
						raise NotImplementedError
						masked_video = video_input * (mask == 0)
						video_clip = auto_inpainting_copy_no_mask(args, video_clip.unsqueeze(0), masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,)
						video_clip_ = video_clip.unsqueeze(0)
						all_video.append(video_clip_[:, 3:])
					video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
					for fps in args.fps_list:
						save_path = args.output_folder
						if not os.path.exists(os.path.join(save_path)):
							os.makedirs(os.path.join(save_path))
						local_save_path = os.path.join(save_path, f'{video_name}.mp4')
						print(f'save in {local_save_path}')
						torchvision.io.write_video(local_save_path, video_, fps=fps)
			


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args = parser.parse_args()
	main(**OmegaConf.load(args.config))


