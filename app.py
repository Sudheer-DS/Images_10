import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("path_to_your_model")
    return model


target_size=(224, 224)
class_names = ['butterfly', 'cat', 'cow', 'dog', 'elefent', 'hen', 'horse', 'sheep', 'spider', 'squirel'] # Replace with actual class names of your model


# Function to preprocess a single image
def preprocess_single_image(image):
    image = image.resize(target_size)  # Resize the image
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to handle predictions for a single image
def handle_single_image(model, uploaded_file):
    try:
        # Open the uploaded file as an image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_image = preprocess_single_image(image)  # Preprocess the image
        predictions = model.predict(input_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(predictions)

        # Display the prediction
        st.write(f"**Predicted Class:** {predicted_class_name}")
        st.write(f"**Confidence:** {confidence:.2f}")
    except Exception as e:
        # Display error message for invalid or corrupted inputs
        st.error("Given input is not valid or Corrupted image")

# Main Streamlit App
def main():
    st.title("Welcome to the Image Prediction App!")
    st.write(
        "This app allows you to upload images and get predictions using a pre-trained model. "
        "We hope you find it helpful!"
    )

    # List model classifications
    st.write("### The model can classify the following types:")
    for idx, class_name in enumerate(class_names):
        st.write(f"{idx + 1}. {class_name}")

    # Input for single images
    st.write("### Upload an Image for Prediction:")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # Load the model
    model = load_model()

    if uploaded_file is not None:
        handle_single_image(model, uploaded_file)
    else:
        st.warning("Please upload a valid image file.")

if __name__ == "__main__":
    main()