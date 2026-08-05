import numpy as np
import tempfile
from PIL import Image
from io import BytesIO

import torch
import torchvision

import streamlit as st
import nst


# set up streamlit interface, similar to setting up a html webpage and css
def interface_setup():
    st.set_page_config(page_title="Style Transfer App", layout="wide", page_icon="🖼")

    # header
    st.title("Neural Style Transfer for Image and Video")
    st.markdown(
        """
        This app functionally combines a content of one image with the style features of another image.    
        Simply upload an image or video and select a style image to begin.
        """
    )
    st.divider()

    # body split into 3 vertical columns
    # col1, _margin1, col2, _margin2, col3 = st.columns([12, 0.3, 12, 0.3, 12])
    col1, col2, col3 = st.columns([12, 12, 12])
    # 0 image, 1 video
    flag = None

    with col1:
        st.subheader("**Upload Your File**")
        st.write("")

        uploaded_file = st.file_uploader(
            " ", type=["png", "jpg", "jpeg", ".mp4"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            ext = uploaded_file.name.split(".")[-1].lower()

            if ext in ["png", "jpg", "jpeg"]:
                flag = 0
                content = Image.open(uploaded_file)
                st.image(content, width="stretch")

            elif ext == "mp4":
                flag = 1
                st.video(uploaded_file, muted=True)

    with col2:
        st.subheader("**Choose Your Style**")
        st.write("")

        sub_col1, sub_col2 = st.columns([2, 1])
        with sub_col1:
            chose_style = st.selectbox(
                " ",
                options=[
                    "Starry Night",
                    "Girl with a Mandolin",
                    "Composition VII",
                    "Indigo Mountains",
                ],
                label_visibility="collapsed",
            )
        with sub_col2:
            style_btn = st.button("Stylize", type="secondary", width="stretch")

        style1 = Image.open("input/gogh.jpg")
        style2 = Image.open("input/picasso.jpg")
        style3 = Image.open("input/kandinsky.jpg")
        style4 = Image.open("input/mountains.jpg")

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.image(style1, caption="Starry Night", width="stretch")
            st.image(style3, caption="Composition VII", width="stretch")
        with sub_col2:
            st.image(style2, caption="Girl with a Mandolin", width="stretch")
            st.image(style4, caption="Indigo Mountains", width="stretch")

    with col3:
        st.subheader("**Your Results**")
        st.write("")

        styles = {
            "Starry Night": "models/parameters_van_gogh.pth",
            "Girl with a Mandolin": "models/parameters_picasso.pth",
            "Composition VII": "models/parameters_kandinsky.pth",
            "Indigo Mountains": "models/parameters_spacefrog.pth",
        }

        if uploaded_file is not None and style_btn:

            if flag == 0:
                output_img = nst.inference(
                    content_image=uploaded_file,
                    checkpoint_model=styles[chose_style],
                )
                st.image(output_img, width="stretch")

                buffer = BytesIO()
                Image.fromarray((output_img * 255).astype(np.uint8)).save(
                    buffer, format="JPEG"
                )
                byte_im = buffer.getvalue()
                st.download_button(
                    label="Download Output Image",
                    data=byte_im,
                    file_name="styled_image.jpg",
                    mime="image/jpg",
                    type="primary",
                )

            elif flag == 1:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                with st.status(
                    f"Processing...",
                    expanded=True,
                ) as status:
                    output_path = "./output/styled_video.mp4"
                    output_vid = nst.inference_video(
                        content_video=temp_file_path,
                        checkpoint_model=styles[chose_style],
                        output_path=output_path,
                    )
                    status.update(label="**Finished!**", state="complete")

                st.video(output_path, muted=True, autoplay=True)

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.download_button(
                    label="Download Output Video",
                    data=video_bytes,
                    file_name=output_path,
                    mime="video/mp4",
                    type="primary",
                    on_click="ignore",
                )

    # footer
    st.divider()
    st.markdown(
        """
        Based on the parametric synthesis algorithm by [Gatys et al. 2016](https://arxiv.org/abs/1508.06576).  
        """
    )


# streamlit run main.py --server.maxUploadSize 400
if __name__ == "__main__":
    interface_setup()
