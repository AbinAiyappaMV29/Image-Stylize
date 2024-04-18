import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

@st.cache(suppress_st_warning=True)  # Cache the function for better performance
def load_image(img):
    # Convert the uploaded image (UploadedFile) to a PIL Image object
    pil_image = Image.open(img)
    
    # Convert PIL Image to numpy array
    img_array = np.array(pil_image)
    
    # Normalize pixel values to range [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension and return as TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_array)[tf.newaxis, ...]
    
    return img_tensor

# Define the Streamlit app
def main():
    st.title("Image Stylization App")

    # Sidebar file uploader for content and style images
    st.sidebar.title("Upload Images")
    content_file = st.sidebar.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'])
    style_file = st.sidebar.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'])

    if content_file and style_file:
        # Display uploaded images
        content_image = load_image(content_file)
        style_image = load_image(style_file)

        # Stylize the content image with the style image
        stylized_image = model(content_image, style_image)[0]

        # Convert the stylized image for display
        stylized_image = np.array(stylized_image) * 255.0
        stylized_image = stylized_image.astype(np.uint8)

        # Display the stylized image
        st.subheader("Stylized Image")
        st.image(stylized_image, caption="Stylized Image", use_column_width=True)

        # Save the stylized image locally
        if st.button("Save Stylized Image"):
            cv2.imwrite('generated_img.jpg', cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR))
            st.success("Stylized image saved successfully!")

if __name__ == '__main__':
    # Run the Streamlit app
    st.set_option('deprecation.showfileUploaderEncoding', False)
    main()
