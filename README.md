# Neural Style Transfer for Video

A user-friendly neural style transfer webapp that aims to stylize both images and video files. Based on [research](https://arxiv.org/abs/1508.06576) of fast feedforward style ML models to significantly improve performance.
Utilizes the [streamlit](https://docs.streamlit.io/) framework as frontend.
<img width="1313" height="713" alt="NST_video_preview" src="https://github.com/user-attachments/assets/f142a0d1-d311-4f9c-a6fe-43fe1a82b56e" />

## Examples

Image Input
![Photo of Tubingen houses](/VST%20Final/input/tubingen.jpg)
Style Input
![The Starry Night by Van Gogh - 1889](/VST%20Final/input/gogh.jpg)
Style Output
![Tubingen stylized with Starry Night](/VST%20Final/output/Starry%20Night_image4.jpg)

Video Input
![Girl looking over waterfall 1080p](/VST%20Final/input/waterfall_1080.mp4)
Style Input
![Composition VII by Wassily Kandinsky - 1913](/VST%20Final/input/kandinsky.jpg)
Style Output
![Waterfall video stylized with Composition VII](/VST%20Final/output/Composition%20VII_video.mp4)

## Usage

Navigate to main directory, run, and open in browser:

```bash
streamlit run main.py

# Local URL: http://localhost:xxxx
```

If you have a powerful Nvidia GPU, download necessary cuDNN and CUDA drivers to speed up rendering time.
Run `cuda.py` to check if viable GPU is enabled.

## Installation

```bash
pip install -r requirements.txt
```
