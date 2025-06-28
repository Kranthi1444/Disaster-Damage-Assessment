import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
from predict import predict_image

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)

state_dict = torch.load("model_aider.pth", map_location=device)
fc_weight_key = 'fc.weight'
fc_bias_key = 'fc.bias'
if (fc_weight_key in state_dict and state_dict[fc_weight_key].shape != model.state_dict()[fc_weight_key].shape):
    state_dict.pop(fc_weight_key)
    state_dict.pop(fc_bias_key)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

def get_accuracy():
    try:
        with open("accuracy.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "N/A"

st.title("Disaster Damage Assessment")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(filepath, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Running prediction..."):
            incident_class, damage_class, repair_cost = predict_image(model, filepath, class_names, device)

    # These must be inside the if-block
        # st.success(f"Incident Type: {incident_class}")
        st.success(f"Prediction: {damage_class}")
        st.info(f"Estimated Repair Cost: {repair_cost}")

accuracy = get_accuracy()
st.write(f"Model Accuracy: {accuracy}")