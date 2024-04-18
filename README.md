# Image-Stylize
"Transform your images with artistic style using this web app powered by TensorFlow and Streamlit. Upload your photos to apply unique artistic filters and save the stylized results with a click!"

# Image Stylization Web App

This web app allows users to apply artistic style transfer to images using TensorFlow and Streamlit. Upload your images, select a style, and instantly generate stylized results!

## Features

- Upload content and style images
- Apply style transfer using a pre-trained TensorFlow model
- View and download the stylized image

## Setup

1. Install dependencies:
   ```bash
   pip install streamlit tensorflow tensorflow-hub opencv-python-headless pillow
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open the web browser and navigate to `http://localhost:8501` to access the app.

## Usage

1. Upload a content image and a style image using the sidebar file uploader.
2. Click the "Stylize" button to apply the selected style to the content image.
3. View the stylized image displayed on the app.
4. Optionally, click "Save Stylized Image" to download the generated stylized image.
