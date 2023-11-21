# LaVie: High-Quality Video Generation with Cascaded Latent Diffusion Models

This repository is the official PyTorch implementation of [LaVie](https://arxiv.org/abs/2309.15103).

**LaVie** is a Text-to-Video (T2V) generation framework, and main part of video generation system [Vchitect](http://vchitect.intern-ai.org.cn/).


[![arXiv](https://img.shields.io/badge/arXiv-2309.15103-b31b1b.svg)](https://arxiv.org/abs/2309.15103)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vchitect.github.io/LaVie-project/)
<!--
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)]()
-->

<img src="lavie.gif" width="800">

## Installation
```
conda env create -f environment.yml 
conda activate lavie
```

## Download Pre-Trained models
Download [pre-trained models](https://huggingface.co/YaohuiW/LaVie/tree/main), [stable diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main), [stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/main) to `./pretrained_models`. You should be able to see the following:
```
├── pretrained_models
│   ├── lavie_base.pt
│   ├── lavie_interpolation.pt
│   ├── lavie_vsr.pt
│   ├── stable-diffusion-v1-4
│   │   ├── ...
└── └── stable-diffusion-x4-upscaler
        ├── ...
```

## Inference
The inference contains **Base T2V**, **Video Interpolation** and **Video Super-Resolution** three steps. We provide several options to generate videos:
* **Step1**: 320 x 512 resolution, 16 frames
* **Step1+Step2**: 320 x 512 resolution, 61 frames
* **Step1+Step3**: 1280 x 2048 resolution, 16 frames
* **Step1+Step2+Step3**: 1280 x 2048 resolution, 61 frames

Feel free to try different options:)


### Step1. Base T2V
Run following command to generate videos from base T2V model. 
```
cd base
python pipelines/sample.py --config configs/sample.yaml
```
Edit `text_prompt` in `configs/sample.yaml` to change prompt, results will be saved under `./res/base`. You may obtain following results. You could also modify `seed` in `configs/sample.yaml` for different results.

<table class="center">
<tr>
  <td><img src="assets/a_Corgi_walking_in_the_park_at_sunrise,_oil_painting_style.gif"></td>
  <td><img src="assets/a_panda_taking_a_selfie,_2k,_high_quality.gif"></td>
  <td><img src="assets/a_polar_bear_playing_drum_kit_in_NYC_Times_Square,_4k,_high_resolution.gif"></td>      
</tr>

<tr>
  <td>a Corgi walking in the park at sunrise, oil painting style</td>
  <td>a panda taking a selfie, 2k, high quality</td>
  <td>a polar bear playing drum kit in NYC Times Square, 4k, high resolution</td>      
</tr>

<tr>
  <td><img src="assets/a_shark_swimming_in_clear_Carribean_ocean,_2k,_high_quality.gif"></td>
  <td><img src="assets/a_teddy_bear_walking_on_the_street,_2k,_high_quality.gif"></td>
  <td><img src="assets/jungle_river_at_sunset,_ultra_quality.gif"></td>
</tr>

<tr>
  <td>a shark swimming in clear Carribean ocean, 2k, high quality</td>
  <td>a teddy bear walking on the street, 2k, high quality</td>
  <td>jungle, river, at sunset, ultra quality</td>
</tr>

</table>


### Step2 (optional). Video Interpolation
Run following command to conduct video interpolation.
```
cd interpolation
python sample.py --config configs/sample.yaml
```
The default input video path is `./res/base`, results will be saved under `./res/interpolation`. In `configs/sample.yaml`, you could modify default `input_folder` with `YOUR_INPUT_FOLDER` in `configs/sample.yaml`. Input videos should be named as `prompt1.mp4`, `prompt2.mp4`, ... and put under `YOUR_INPUT_FOLDER`. Launching the code will process all the input videos in `input_folder`.

<table class="center">
<tr>
  <td><img src="assets/interpolation/a_teddy_bear_walking_on_the_street,_2k,_high_quality.gif"></td>
  <td><img src="assets/interpolation/a_Corgi_walking_in_the_park_at_sunrise,_oil_painting_style.gif"></td>
  <td><img src="assets/interpolation/a_panda_taking_a_selfie,_2k,_high_quality.gif"></td>   
</tr>
        
<tr>
  <td>a teddy bear walking on the street, 2k, high_quality</td>
  <td>a Corgi walking in the park at sunrise, oil painting style</td>
  <td>a panda taking a selfie, 2k, high quality</td>   
</tr>

</table>

### Step3 (optional). Video Super-Resolution
Run following command to conduct video super-resolution.
```
cd vsr
python sample.py --config configs/sample.yaml
```
The default input video path is `./res/base` and results will be saved under `./res/vsr`. You could modify default `input_path` with `YOUR_INPUT_FOLDER` in `configs/sample.yaml`. Smiliar to Step2, input videos should be named as `prompt1.mp4`, `prompt2.mp4`, ... and put under `YOUR_INPUT_FOLDER`. Launching the code will process all the input videos in `input_folder`.

<table class="center">
        <tr>
        <td><img src="assets/vsr/a_shark_swimming_in_clear_Carribean_ocean,_2k,_high_quality.gif"></td>
        </tr>
        <tr>
        <td>a shark swimming in clear Carribean ocean, 2k, high quality</td>
        </tr>
</table>

## BibTex
```bibtex
@article{wang2023lavie,
  title={LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models},
  author={Wang, Yaohui and Chen, Xinyuan and Ma, Xin and Zhou, Shangchen and Huang, Ziqi and Wang, Yi and Yang, Ceyuan and He, Yinan and Yu, Jiashuo and Yang, Peiqing and others},
  journal={arXiv preprint arXiv:2309.15103},
  year={2023}
}
```

## Disclaimer
We disclaim responsibility for user-generated content. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities. It is prohibited for pornographic, violent and bloody content generation, and to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

## Contact Us
**Yaohui Wang**: [wangyaohui@pjlab.org.cn](mailto:wangyaohui@pjlab.org.cn)  
**Xinyuan Chen**: [chenxinyuan@pjlab.org.cn](mailto:chenxinyuan@pjlab.org.cn)  
**Xin Ma**: [xin.ma1@monash.edu](mailto:xin.ma1@monash.edu)

## Acknowledgements
The code is built upon [diffusers](https://github.com/huggingface/diffusers) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion), we thank all the contributors for open-sourcing. 


## License
The code is licensed under Apache-2.0, model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please contact vchitect@pjlab.org.cn.
