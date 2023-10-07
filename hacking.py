import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import torch
from fastai.vision.all import *
from fastai.metrics import *



# Load your model here (replace with your own model)
# model = MyCNN()
# checkpoint = torch.load("my_cnn_model.pth",map_location=torch.device("cpu"))
# model.load_state_dict(checkpoint['model_state_dict'])
#model = torch.load("my_cnn_model.pth",map_location=torch.device("cpu"))

def GetLabel(fileName):
  return fileName.split('_')[0]

AllImagesDir = 'C:\\Users\\Niv Doshi\\Desktop\\Jupyter\\Hackathon\\RiceDiseaseDataset\\train\\AllImages'
dls = ImageDataLoaders.from_name_func(
  AllImagesDir, get_image_files(AllImagesDir), valid_pct=0.2, seed=420,
  label_func=GetLabel, item_tfms=Resize(224))

model = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=True)
model = model.load('C:\\Users\\Niv Doshi\\Desktop\\hacking\\my_cnn_model')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

# Function to make predictions using your model
def predict_image(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction

# Streamlit app
st.set_page_config(
    page_title="Image Classification App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
custom_css = """
<style>
.stButton>button {
    background-color: #007BFF !important;
    color: white !important;
    border-color: #007BFF !important;
}
.stButton>button:hover {
    background-color: #0056b3 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "About Us"])

# Toggle for dark mode
#dark_mode = st.sidebar.checkbox("Dark Mode")

# if dark_mode:
#     # Apply dark mode theme
#     st.markdown(
#         """
#         <style>
#         body {
#             background-color: #121212 !important;
#             color: white !important;
#         }
#         .stSidebar {
#             background-color: #1E1E1E !important;
#             color: white !important;
#         }
#         .stButton {
#             background-color: #007BFF !important;
#             color: white !important;
#             border-color: #007BFF !important;
#         }
#         .stButton:hover {
#             background-color: #0056b3 !important;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
predicted_class = ""
if page == "Home":
    st.header("Welcome to the Image Classification App")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            image = PILImage.create(uploaded_image)
            image.show()
            label,_,probs = model.predict(image)
            
            
            # # Display the prediction
            st.subheader("Prediction:")
            predicted_class = label

            st.write(f"Class: {predicted_class}")
            #st.write(f"Confidence: {confidence:.2f}")

    # Create a section to display reports
    st.header("Report")
    if (predicted_class == "LeafBlast"):
        st.write("Leaf blast typically starts as small, water-soaked lesions on rice leaves. As the disease progresses, these lesions expand and develop into spindle-shaped or diamond-shaped lesions with gray centers and dark brown to black borders. Severely affected leaves may become necrotic (die) and eventually wither and die.In addition to leaves, the disease can also affect other plant parts, including stems, nodes, panicles, and even the rice grains.")
    elif (predicted_class == "BrownSpot"):
        st.write("Small oval to elliptical brown to black spots. Yellow halos around the spots. Lesions may enlarge and coalesce. Severe infections can lead to leaf blighting and reduced yield. Warm and humid conditions favor its development.")
    elif (predicted_class == "Healthy"):
        st.write("Healthy rice leaf ")
    elif (predicted_class == "Hispa"):
        st.write("Windowpane damage: Transparent patches on leaves due to feeding. Skeletonization: Green tissue between veins eaten, leaving a lacy appearance. Leaf curling: Leaves may curl or roll inward in response to damage. Stunted growth: Severe infestations can lead to stunted plant growth. Reduced yield: Loss of leaf area can result in lower rice yields.")
    else:
        st.write("Default")

    st.header("Treatment")
    if (predicted_class == "LeafBlast"):
        st.write("Treatment for LeafBlast")
    elif (predicted_class == "BrownSpot"):
        st.write("Treatment for BrownSpot")
    elif (predicted_class == "Healthy"):
        st.write("Healthy rice leaf ")
    elif (predicted_class == "Hispa"):
        st.write("Treatment for Hispa")
    else:
        st.write("Default") 
    
    # You can add more text, charts, or other report elements here.

elif page == "About Us":
    st.header("About Us")
    
    # Custom styling for the "About Us" section
    about_us_style = """
    <style>
    .stHeader {
        background-color: #007BFF;
        color: white;
        padding: 10px;
    }
    </style>
    """
    st.markdown(about_us_style, unsafe_allow_html=True)
    
    st.write("This is the about us page.")
    st.write("We are a team of developers working on this image classification app.")

st.sidebar.title("About")
st.sidebar.info(
    "This is a simple image classification app using Streamlit and TensorFlow."
)
