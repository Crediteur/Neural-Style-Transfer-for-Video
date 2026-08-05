# Neural Style Transfer for Video

A user-friendly photo filter app based on Neural Style Transfer research by [Gatys](https://arxiv.org/abs/1508.06576). Expands original functionality to include video file processing by using masking techniques to preserve continuous motion and reduce artifacts. Exchangeable weights from pre-training on specific images allow fast feed forward style processing to significantly improve computation performance- at the trade-off of limited style options.
Utilizes [Streamlit](https://docs.streamlit.io/) framework as a UX frontend. 

Try the online version [here](https://neural-style-transfer-for-video-nbjk7qgsj6tebyvdbhlu2q.streamlit.app/). Note this is the slower, CPU version, which may not work well when rendering videos.

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
      <tr>
      <td>
        <img 
          src="https://github.com/user-attachments/assets/27e2c3c1-ee9d-4899-a804-654cc09d6244" 
          alt="Photo of family" 
          width="400">
      </td>
      <td>
        <img 
          src="VST%20Final/input/kandinsky.jpg" 
          alt="Composition VII by Kandinsky" 
          width="400">
      </td>
      <td>
        <img 
          src="https://github.com/user-attachments/assets/ace792a7-e79b-42d7-a585-2210d3225a1b" 
          alt="Family photo styled in Kandinsky's style" 
          width="400">
      </td>
    </tr>
    <tr>
      <td>
        <img 
          src="https://github.com/user-attachments/assets/a97e367b-a7f4-4dad-a386-5cb9bfea1440" 
          alt="Photo of peaches in a vase stand" 
          width="400">
      </td>
      <td>
        <img 
          src="VST%20Final/input/picasso.jpg" 
          alt="The Starry Night by Van Gogh" 
          width="400">
      </td>
      <td>
        <img 
          src="https://github.com/user-attachments/assets/08cbf979-72bd-4a5f-853d-1be0d3722ab1" 
          alt="Picasso styled peaches" 
          width="400">
      </td>
    </tr>
</table>

<table>
  <tr>
    <td>
      <video 
        src="https://github.com/user-attachments/assets/fb91116f-5d85-431e-b0d4-3e710bbb814e"
        alt="Video of a girl looking over a waterfall 1080p">
    </td>
    <td>
      <video 
        src="https://github.com/user-attachments/assets/140213da-8c57-4649-9df3-ee740c3f0d4a"
        alt="Kandinsky styled video of a girl looking over a waterfall">
    </td>
  </tr>
  <tr>
    <td>
      <video 
        src="https://github.com/user-attachments/assets/bd6de812-7744-46c2-b892-b0d2b08f7e00"
        alt="Kandinsky styled video of a girl looking over a waterfall">
    </td>
    <td>
      <video 
        src="https://github.com/user-attachments/assets/2a47ac8a-9b92-4b41-b217-f0e09cb4b2be"
        alt="">
    </td>
  </tr>
</table>

## Usage

Navigate to main directory, run, and open in browser:

```bash
streamlit run main.py

# Local URL: http://localhost:xxxx
```

If you have a compatible Nvidia GPU, download necessary cuDNN and CUDA drivers to speed up rendering time.
Run `cuda.py` to check if viable GPU is enabled.

## Installation

```bash
pip install -r requirements.txt
```
