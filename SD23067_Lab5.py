# ---------------------------------
# Image Classification Web App
# Using PyTorch + Streamlit (CPU)
# ---------------------------------

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ---------------------------------
# Step 1: Page configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Image Recognition Demo",
    layout="centered"
)

st.title("AI Image Recognition System")
st.write(
    "This application uses a **pre-trained ResNet18 model** to classify images "
    "into ImageNet categories using CPU-based inference."
)

# ---------------------------------
# Step 2 & 3: Device configuration
# ---------------------------------
device = torch.device("cpu")

# ---------------------------------
# Step 4: Load pre-trained model
# ---------------------------------
@st.cache_resource
def load_resnet_model():
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.eval()
    net.to(device)
    return net

model = load_resnet_model()

# ---------------------------------
# Step 5: Image preprocessing
# ---------------------------------
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet labels
imagenet_labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

# ---------------------------------
# Step 6: Image upload UI
# ---------------------------------
uploaded_image = st.file_uploader(
    "Select an image file (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:
    # Display image
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    # ---------------------------------
    # Step 7: Preprocess & inference
    # ---------------------------------
    img_tensor = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    # ---------------------------------
    # Step 8: Softmax & Top-5 classes
    # ---------------------------------
    probs = torch.softmax(prediction[0], dim=0)
    top_probs, top_indices = torch.topk(probs, 5)

    # Prepare results table
    output_data = {
        "Predicted Class": [imagenet_labels[i] for i in top_indices],
        "Confidence (%)": [round(p.item() * 100, 2) for p in top_probs]
    }

    df_results = pd.DataFrame(output_data)

    st.subheader("Top 5 Classification Results")
    st.dataframe(df_results, use_container_width=True)

    # ---------------------------------
    # Step 9: Bar chart visualization
    # ---------------------------------
    st.subheader("Prediction Confidence Comparison")
    st.bar_chart(
        df_results.set_index("Predicted Class")["Confidence (%)"]
    )

else:
    st.info("Please upload an image to perform classification.")

