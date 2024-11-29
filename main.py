import streamlit as st
import numpy as np
import cv2
import easyocr
from PIL import Image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    preprocessed = cv2.medianBlur(thresh, 3)
    return preprocessed

def extract_text(image, languages=['en', 'ar']):
    reader = easyocr.Reader(languages, gpu=True)
    result = reader.readtext(image, detail=0)
    return result

st.title("OCR Tool with Preprocessing")
st.write("Upload or scan an image to extract text.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

use_camera = st.checkbox("Use Camera")

image_to_process = None

if uploaded_file is not None:
    image_to_process = np.array(Image.open(uploaded_file))
    st.image(image_to_process, caption="Uploaded Image", use_column_width=True)

elif use_camera:
    picture = st.camera_input("Take a picture")
    if picture:
        image_to_process = np.array(Image.open(picture))
        st.image(image_to_process, caption="Captured Image", use_column_width=True)

if image_to_process is not None:
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR))

    preprocessed_image = preprocess_image(temp_image_path)
    cv2.imwrite("preprocessed_image.png", preprocessed_image)

    st.image(preprocessed_image, caption="Preprocessed Image", channels="GRAY")

    st.write("Recognized Text:")
    recognized_text = extract_text("preprocessed_image.png")
    for line in recognized_text:
        st.write(line)
