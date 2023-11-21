# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import json
import numpy as np
import torch
import shutil
from omegaconf import OmegaConf
import imageio
from einops import rearrange
import torchvision
from torchvision import transforms
from decord import VideoReader
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from cog import BasePredictor, Input, Path

# base model
sys.path.insert(0, "base")
sys.path.insert(0, "base/pipelines")
from base.models import get_models as get_models_base
from download import find_model
from pipeline_videogen import VideoGenPipeline

# super resolution
sys.path.insert(0, "vsr")
from vsr.models.autoencoder_kl import AutoencoderKL
from vsr.models.unet import UNet3DVSRModel
from vsr.models.pipeline_stable_diffusion_upscale_video_3d import (
    StableDiffusionUpscalePipeline,
)

# interpolation model
sys.path.insert(0, "interpolation")
from interpolation.models import get_models as get_models_interpolation
from interpolation.datasets import video_transforms
from interpolation.models.clip import TextEmbedder
from interpolation.diffusion import create_diffusion


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pretrained_path = "pretrained_models"
        torch.set_grad_enabled(False)
        self.device = "cuda:0"

        #  ============ base model ============
        args_base = OmegaConf.load("base/configs/sample.yaml")
        sd_path = pretrained_path + "/stable-diffusion-v1-4"
        self.unet = get_models_base(args_base, sd_path).to(
            self.device, dtype=torch.float16
        )
        state_dict = find_model(pretrained_path + "/lavie_base.pt")
        self.unet.load_state_dict(state_dict)

        self.vae = AutoencoderKL.from_pretrained(
            sd_path, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            sd_path, subfolder="tokenizer"
        )
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device)

        self.unet.eval()
        self.vae.eval()
        self.text_encoder_one.eval()

        self.schedulers = {
            "ddim": DDIMScheduler.from_pretrained(
                sd_path,
                subfolder="scheduler",
                beta_start=args_base.beta_start,
                beta_end=args_base.beta_end,
                beta_schedule=args_base.beta_schedule,
            ),
            "eulerdiscrete": EulerDiscreteScheduler.from_pretrained(
                sd_path,
                subfolder="scheduler",
                beta_start=args_base.beta_start,
                beta_end=args_base.beta_end,
                beta_schedule=args_base.beta_schedule,
            ),
            "ddpm": DDPMScheduler.from_pretrained(
                sd_path,
                subfolder="scheduler",
                beta_start=args_base.beta_start,
                beta_end=args_base.beta_end,
                beta_schedule=args_base.beta_schedule,
            ),
        }

        # ============ interpolation model ============
        interpolation_ckpt_path = pretrained_path + "/lavie_interpolation.pt"
        self.args_interpolation = OmegaConf.load(
            "interpolation/configs/sample.yaml"
        ).args

        self.interpolation_model = get_models_interpolation(
            self.args_interpolation,
            sd_path,
        ).to(self.device)
        self.interpolation_model.enable_xformers_memory_efficient_attention()

        print(f"Loading interpolation model from {interpolation_ckpt_path}.")
        state_dict = find_model(interpolation_ckpt_path)
        self.interpolation_model.load_state_dict(state_dict)
        self.interpolation_model.eval()
        self.text_encoder = TextEmbedder(sd_path).to(self.device)
        self.diffusion = create_diffusion(str(50))
        self.vae_full = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(
            self.device
        )

        # ============ super resolution ============
        self.args_sr = OmegaConf.load("vsr/configs/sample.yaml")
        self.sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            pretrained_path + "/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
        )
        # vae
        self.sr_pipeline.vae = AutoencoderKL.from_config("vsr/configs/vae_config.json")
        sr_pretrained_model = (
            pretrained_path
            + "/stable-diffusion-x4-upscaler/vae/diffusion_pytorch_model.bin"
        )
        self.sr_pipeline.vae.load_state_dict(
            torch.load(sr_pretrained_model, map_location="cpu")
        )
        # unet
        config_path = "vsr/configs/unet_3d_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        config["video_condition"] = False
        self.sr_pipeline.unet = UNet3DVSRModel.from_config(config)

        sr_lavie = pretrained_path + "/lavie_vsr.pt"
        print(f"Loading super resolution model from {sr_lavie}.")
        sr_checkpoint = torch.load(sr_lavie, map_location="cpu")["ema"]
        self.sr_pipeline.unet.load_state_dict(sr_checkpoint, True)
        self.sr_pipeline.unet = self.sr_pipeline.unet.half()
        self.sr_pipeline.unet.eval()  # important!

        # DDIMScheduler
        with open(
            pretrained_path
            + "/stable-diffusion-x4-upscaler/scheduler/scheduler_config.json",
            "r",
        ) as f:
            config = json.load(f)
        config["beta_schedule"] = "linear"
        self.sr_pipeline.scheduler = DDIMScheduler.from_config(config)
        self.sr_pipeline = self.sr_pipeline.to(self.device)

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for video generation.",
            default="a Corgi walking in the park at sunrise, oil painting style",
        ),
        sample_method: str = Input(
            description="Choose a scheduler for sampling base video output.",
            default="ddpm",
            choices=["ddim", "eulerdiscrete", "ddpm"],
        ),
        width: int = Input(description="Width of output video.", default=512),
        height: int = Input(description="Height of output video", default=320),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", default=7.0
        ),
        quality: int = Input(
            description="Quality of the output vide0", le=10, ge=0, default=9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        interpolation: bool = Input(
            description="Default output has 16 frames. Set interpolation to True to get 61 frames output.",
            default=False,
        ),
        super_resolution: bool = Input(
            description="Super resolution 4x when set to True.", default=False
        ),
        video_fps: int = Input(
            description="Number of frames per second in the output video", default=8
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        temp_output_dir = "temp_output"
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        os.makedirs(temp_output_dir)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        scheduler = self.schedulers[sample_method]

        videogen_pipeline = VideoGenPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            tokenizer=self.tokenizer_one,
            scheduler=scheduler,
            unet=self.unet,
        ).to(self.device)
        videogen_pipeline.enable_xformers_memory_efficient_attention()

        videos = videogen_pipeline(
            prompt,
            video_length=16,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).video

        base_video_path = f"{temp_output_dir}/base.mp4"
        imageio.mimwrite(base_video_path, videos[0], fps=video_fps, quality=quality)
        output_path = base_video_path
        torch.cuda.empty_cache()

        if interpolation:
            print("Running interpolation...")
            self.args_interpolation.image_h = height
            self.args_interpolation.image_w = width
            self.args_interpolation.latent_h = height // 8
            self.args_interpolation.latent_w = width // 8

            video_input = get_input(self.args_interpolation, base_video_path)  # f,c,h,w
            print(video_input.shape)
            video_input = video_input.to(self.device).unsqueeze(0)  # b,f,c,h,w
            all_video = [video_input]

            video_clip = auto_inpainting_copy_no_mask(
                self.args_interpolation,
                video_input,
                prompt + self.args_interpolation.additional_prompt,
                self.vae_full,
                self.text_encoder,
                self.diffusion,
                self.interpolation_model,
                self.device,
            )
            all_video.append(video_clip.unsqueeze(0))
            output_video = (
                ((video_clip * 0.5 + 0.5) * 255)
                .add_(0.5)
                .clamp_(0, 255)
                .to(dtype=torch.uint8)
                .cpu()
                .permute(0, 2, 3, 1)
            )
            interpolation_video_path = f"{temp_output_dir}/interpolation.mp4"
            torchvision.io.write_video(
                interpolation_video_path, output_video, fps=video_fps
            )
            output_path = interpolation_video_path
            torch.cuda.empty_cache()

        if super_resolution:
            print("Running super resolution...")
            lr_path = output_path
            negative_prompt = "blur, worst quality"

            vframes, aframes, info = torchvision.io.read_video(
                filename=lr_path, pts_unit="sec", output_format="TCHW"
            )  # RGB
            vframes = vframes / 255.0
            vframes = (vframes - 0.5) * 2  # T C H W [-1, 1]
            t, _, h, w = vframes.shape
            vframes = vframes.unsqueeze(dim=0)  # 1 T C H W
            vframes = rearrange(
                vframes, "b t c h w -> b c t h w"
            ).contiguous()  # 1 C T H W
            print(
                "LR video Input_shape:",
                vframes.shape,
                "Noise_level:",
                self.args_sr.noise_level,
                "Guidance_scale:",
                self.args_sr.guidance_scale,
            )
            generator = torch.Generator(device=self.device).manual_seed(seed)

            with torch.no_grad():
                short_seq = 8
                vframes_seq = vframes.shape[2]
                if vframes_seq > short_seq:  # for VSR
                    upscaled_video_list = []
                    for start_f in range(0, vframes_seq, short_seq):
                        print(
                            f"Processing: [{start_f}-{start_f + short_seq}/{vframes_seq}]"
                        )
                        torch.cuda.empty_cache()  # delete for VSR
                        end_f = min(vframes_seq, start_f + short_seq)
                        upscaled_video_ = self.sr_pipeline(
                            prompt,
                            image=vframes[:, :, start_f:end_f],
                            generator=generator,
                            num_inference_steps=self.args_sr.inference_steps,
                            guidance_scale=self.args_sr.guidance_scale,
                            noise_level=self.args_sr.noise_level,
                            negative_prompt=negative_prompt,
                        ).images  # T C H W [-1, 1]
                        upscaled_video_list.append(upscaled_video_)
                    upscaled_video = torch.cat(upscaled_video_list, dim=0)
                else:
                    upscaled_video = self.sr_pipeline(
                        prompt,
                        image=vframes,
                        generator=generator,
                        num_inference_steps=self.args_sr.inference_steps,
                        guidance_scale=self.args_sr.guidance_scale,
                        noise_level=self.args_sr.noise_level,
                        negative_prompt=negative_prompt,
                    ).images  # T C H W [-1, 1]

                print("Output of the super resolution:", upscaled_video.shape)
                upscaled_video = (upscaled_video / 2 + 0.5).clamp(0, 1) * 255
                upscaled_video = upscaled_video.permute(0, 2, 3, 1).to(torch.uint8)
                upscaled_video = upscaled_video.numpy().astype(np.uint8)
                sr_output_path = f"{temp_output_dir}/sr.mp4"
                imageio.mimwrite(
                    sr_output_path, upscaled_video, fps=video_fps, quality=quality
                )
                output_path = sr_output_path

        final_output_path = "/tmp/out.mp4"
        shutil.copy(output_path, final_output_path)
        return Path(final_output_path)


def get_input(args, input_path):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),
            video_transforms.ResizeVideo((args.image_h, args.image_w)),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    temporal_sample_func = video_transforms.TemporalRandomCrop(
        args.num_frames * args.frame_interval
    )
    video_reader = VideoReader(input_path)
    total_frames = len(video_reader)
    start_frame_ind, end_frame_ind = temporal_sample_func(total_frames)
    frame_indice = np.linspace(
        start_frame_ind, end_frame_ind - 1, args.num_frames, dtype=int
    )
    video_frames = (
        torch.from_numpy(video_reader.get_batch(frame_indice).asnumpy())
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    video_frames = transform_video(video_frames)
    return video_frames


def auto_inpainting_copy_no_mask(
    args, video_input, prompt, vae, text_encoder, diffusion, model, device
):
    b, f, c, h, w = video_input.shape
    video_input = rearrange(video_input, "b f c h w -> (b f) c h w").contiguous()
    video_input = vae.encode(video_input).latent_dist.sample().mul_(0.18215)
    video_input = rearrange(video_input, "(b f) c h w -> b c f h w", b=b).contiguous()

    lr_indice = torch.IntTensor([i for i in range(0, 62, 4)]).to(device)
    copied_video = torch.index_select(video_input, 2, lr_indice)
    copied_video = torch.repeat_interleave(copied_video, 4, dim=2)
    copied_video = copied_video[:, :, 1:-2, :, :]
    copied_video = (
        torch.cat([copied_video] * 2)
        if args.do_classifier_free_guidance
        else copied_video
    )

    z = torch.randn(
        1, 4, args.num_frames, args.latent_h, args.latent_w, device=device
    )  # b,c,f,h,w
    z = torch.cat([z] * 2) if args.do_classifier_free_guidance else z
    prompt_all = (
        [prompt] + [args.negative_prompt]
        if args.do_classifier_free_guidance
        else [prompt]
    )
    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, class_labels=None)

    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
        mask=None,
        x_start=copied_video,
        use_concat=args.use_concat,
        copy_no_mask=args.copy_no_mask,
    )

    samples, _ = samples.chunk(2, dim=0)  # [1, 4, 16, 32, 32]
    video_clip = samples[0].permute(1, 0, 2, 3).contiguous()  # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample  # [16, 3, 256, 256]
    return video_clip
