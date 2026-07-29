# Neural Style Transfer for Video

A user-friendly neural style transfer webapp that aims to stylize both images and video files. Based on [research](https://arxiv.org/abs/1508.06576) of fast feedforward style ML models to significantly improve performance.
Utilizes the [streamlit](https://docs.streamlit.io/) framework as frontend.
<img width="1313" height="713" alt="NST_video_preview" src="https://github.com/user-attachments/assets/f142a0d1-d311-4f9c-a6fe-43fe1a82b56e" />

## Examples

<table>
  <tr>
    <th>Image Input</th>
    <th>Style</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>
      <img 
        src="VST%20Final/input/tubingen.jpg" 
        alt="Photo of Tubingen houses" 
        width="400">
    </td>
    <td>
      <img 
        src="VST%20Final/input/gogh.jpg" 
        alt="The Starry Night by Van Gogh" 
        width="400">
    </td>
    <td>
      <img 
        src="VST%20Final/output/Starry%20Night_image4.jpg" 
        alt="Tubingen stylized with Starry Night" 
        width="400">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th>Video Input</th>
    <th>Style</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>
      <video 
        src="https://github.com/user-attachments/assets/fb91116f-5d85-431e-b0d4-3e710bbb814e"
        alt="video of a girl looking over a waterfall 1080p"
        width="100" height="100">
    </td>
    <td>
      <img 
        src="VST%20Final/input/kandinsky.jpg" 
        alt="Composition VII by Wassily Kandinsky" 
        width="400">
    </td>
    <td>
      <video 
        src="https://github.com/user-attachments/assets/bd6de812-7744-46c2-b892-b0d2b08f7e00"
        alt="Kandinsky styled video of a girl looking over a waterfall"
        width="100" height="100">
    </td>
  </tr>
</table>

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
