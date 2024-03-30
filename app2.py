import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import gradio as gr
import cv2

# Load YOLOv5 model
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

# Streamlit App
st.title("YOLOv5 Fire Detection Demo")
st.write("Choose an option below:")

# Option to upload an image
uploaded_file = st.file_uploader("Option 1: Upload an image", type=["jpg", "jpeg", "png"])

# Option to capture image through webcam
if st.button("Option 2: Capture image from Webcam"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(image, caption="Captured Image.", use_column_width=True)

        # Perform YOLOv5 inference
        result_image = yolo(image)

        # Display result image
        st.image(result_image, caption="Processed Image.", use_column_width=True)
    else:
        st.warning("Unable to capture image from webcam. Please make sure your webcam is connected.")

# If an image is uploaded, perform YOLOv5 inference
if uploaded_file is not None:
    # Read image and display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Perform YOLOv5 inference
    result_image = yolo(image)

    # Display result image
    st.image(result_image, caption="Processed Image.", use_column_width=True)
