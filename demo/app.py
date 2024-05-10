import gradio as gr
import torch
from PIL import Image, ImageDraw
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', 'demo/best.pt')  # force_reload=True to update

def yolo(im):
    # Convert PIL Image to NumPy array
    im_np = np.array(im)

    # Inference
    results = model(im_np)

    # Accessing bounding box information
    bboxes = results.xyxy[0].cpu().numpy()

    # Drawing bounding boxes on the image
    draw = ImageDraw.Draw(im)
    for bbox in bboxes:
        label = int(bbox[-1])
        draw.rectangle(bbox[:4], outline="red", width=3)
        draw.text((bbox[0], bbox[1]), f"Class: {label}", fill="red")

    return im

title = "YOLOv5"
description = "YOLOv5 demo for fire detection. Upload an image or click an example image to use."
article = "See https://github.com/robmarkcole/fire-detection-from-images"
examples = [['demo/pan-fire.jpg'], ['demo/fire-basket.jpg']]

gr.Interface(yolo, 
             inputs=gr.Image(type='pil', label="Original Image"),
             outputs=gr.Image(type="pil", label="Output Image"),
             title=title,
             description=description,
             article=article,
             examples=examples).launch(debug=True)
