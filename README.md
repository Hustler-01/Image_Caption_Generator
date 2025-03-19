# Image_Caption_Generator 🖼️
This project demonstrates an end-to-end system for generating captions for images. The pipeline involves:
- Using the **VGG16** model for feature extraction.
- Building and training a **combination of LSTM and a neural network** for generating captions.
- Evaluating the model's performance using the **BLEU score**.
- Converting the generated captions to speech using the **gTTS** library.
- Developing a user-friendly frontend with Streamlit for interacting with the model.

---
## Deployment
This project is deployed using streamlit.
https://huggingface.co/spaces/AzmeenKhatoon/Image_Caption_Generator

## Dataset 📊
This project uses the **Flickr8k dataset**, which contains 8,000 images with five captions each, available on [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k). Download the dataset and prepare it for training by extracting the images and captions.

---
