import streamlit as st 
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from gtts import gTTS
import time

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)  # Extract last CNN layer

# Load trained captioning model
model = load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Set Streamlit page config
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️", layout="wide")

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    body { background-color: #1E1E1E; color: white; }
    .stButton>button { background-color: #ff7f50; color: white; border-radius: 8px; padding: 10px; width: 100%; }
    .stTextInput>div>div>input { background-color: #262730; color: white; }
    .stMarkdown { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App Title
st.markdown("<h1 style='text-align: center; color: #ff7f50;'>AI Image Caption Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Upload an image, and our AI will generate a caption for it.</p>", unsafe_allow_html=True)

# File uploader
uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

# Function to get word from index
def get_word_from_index(index, tokenizer):
    return tokenizer.index_word.get(index, None)  # Faster lookup

# Function to generate caption
def predict_caption(model, image_features, tokenizer, max_caption_length=35):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)

        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)

        if predicted_word is None or predicted_word == "endseq":
            break
        caption += " " + predicted_word

    return caption.replace("startseq", "").replace("endseq", "").strip()

# Process uploaded image
if uploaded_image is not None:
    col1, col2 = st.columns([2, 3])  # Two-column layout for better UI
    
    with col1:
        st.image(uploaded_image, use_container_width=True, caption="📷 Uploaded Image")
    
    with col2:
        with st.spinner("⏳ Generating caption..."):
            time.sleep(1)  # Simulate loading effect
            
            # Convert uploaded image to PIL format
            uploaded_image = Image.open(uploaded_image)

            # Preprocess the image
            image = uploaded_image.resize((224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image = preprocess_input(image)

            # Extract features using VGG16
            image_features = vgg_model.predict(image, verbose=0)
            image_features = image_features.reshape((1, -1))  # Flatten for LSTM input

            # Generate caption
            generated_caption = predict_caption(model, image_features, tokenizer)

            # Display generated caption
            st.markdown(f"<h3 style='color: #ffcc00;'>📝 Generated Caption:</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 20px; color: #000000;'><b>{generated_caption}</b></p>", unsafe_allow_html=True)

            # Convert caption to speech
            tts = gTTS(generated_caption, lang="en")
            audio_path = "predicted_caption.mp3"
            tts.save(audio_path)

            st.audio(audio_path, format="audio/mp3")
