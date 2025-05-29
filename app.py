
import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
from predict import predict_image
# Setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels (4 classes)
class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

# Load model with output size = 4
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)  # output layer with 4 classes

# Load weights with care to avoid mismatch
state_dict = torch.load("model_aider.pth", map_location=device)

# Filter out FC layer weights if size mismatch
fc_weight_key = 'fc.weight'
fc_bias_key = 'fc.bias'

if (fc_weight_key in state_dict and 
    state_dict[fc_weight_key].shape != model.state_dict()[fc_weight_key].shape):
    # Remove fc weights and bias from the loaded dict to avoid mismatch
    state_dict.pop(fc_weight_key)
    state_dict.pop(fc_bias_key)

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()


# Function to get accuracy text
def get_accuracy():
    try:
        with open("accuracy.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "N/A"

# Define estimation costs for each damage class
estimation_costs = {
    'no-damage': '0 USD',
    'minor-damage': '500 - 2000 USD',
    'major-damage': '2000 - 10000 USD',
    'destroyed': 'Above 10000 USD'
}

# Streamlit app
st.title("Disaster Damage Assessment")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(filepath, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Running prediction..."):
            prediction = predict_image(model, filepath, class_names, device)
        st.success(f"Prediction: {prediction}")
        
        # Show estimated cost based on prediction
        estimated_cost = estimation_costs.get(prediction, "Cost estimation not available")
        st.write(f"Estimated Repair Cost: **{estimated_cost}**")


accuracy = get_accuracy()
st.write(f"Model Accuracy: {accuracy}")


# from flask import Flask, render_template, request
# import torch
# import torchvision.models as models
# import torch.nn as nn
# from PIL import Image
# from torchvision import transforms
# import os
# from predict import predict_image
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Class labels (4 classes)
# class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

# # Load model
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 4)  # output layer for 4 classes

# # Load weights safely
# state_dict = torch.load("model_aider.pth", map_location=device)
# fc_weight_key = 'fc.weight'
# fc_bias_key = 'fc.bias'

# if (fc_weight_key in state_dict and
#     state_dict[fc_weight_key].shape != model.state_dict()[fc_weight_key].shape):
#     state_dict.pop(fc_weight_key)
#     state_dict.pop(fc_bias_key)

# model.load_state_dict(state_dict, strict=False)
# model.to(device)
# model.eval()



# # Function to read accuracy
# def get_accuracy():
#     try:
#         with open("accuracy.txt", "r") as f:
#             return f.read().strip()
#     except FileNotFoundError:
#         return "N/A"

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     image_url = None
#     if request.method == "POST":
#         file = request.files.get("file")
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             prediction = predict_image(model, filepath, class_names, device)
#             image_url = filepath

#     accuracy = get_accuracy()
#     return render_template("index.html", prediction=prediction, image_url=image_url, accuracy=accuracy)

# if __name__ == "__main__":
#     app.run(debug=True)

