import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import gen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def test_transform(image_size=None):
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(
        resize + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return transform


# postprocess the images for saving
def denormalize(tensors):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors


# prepare image by colour and dimension normalization
def preprocess_frame(frame):
    # convert colour to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    tensor_rgb = transform(frame_rgb).unsqueeze(0)
    return tensor_rgb.to(device)


# process the image data
def postprocess_frame(tensor):
    img_np = tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_bgr


# get video data
def get_video_info(video):
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = total_frame / fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    return fps, length, total_frame, w, h


def inference(content_image, checkpoint_model):
    # load model
    transformer = gen.TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()

    # inference
    image_tensor = Variable(test_transform()(Image.open(content_image))).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    stylized_image_np = stylized_image.squeeze(0).permute(1, 2, 0).numpy()
    stylized_image_np = np.clip(stylized_image_np, 0, 1)
    return stylized_image_np


def inference_video(content_video, checkpoint_model, output_path):
    # load model
    transformer = gen.TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()

    # video attributes
    cap = cv2.VideoCapture(content_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # set up mp4 codec and writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # inference
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tensor_rgb = preprocess_frame(frame)
        with torch.no_grad():
            stylized_tensor = transformer(tensor_rgb)
        stylized_tensor = denormalize(stylized_tensor).cpu()
        stylized_frame = postprocess_frame(stylized_tensor)
        out.write(stylized_frame)

    # release
    cap.release()
    out.release()
