import torch
from model import DamageClassifier
from predict import predict_image  # your prediction function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']

model = DamageClassifier(num_classes=4)

# Load the checkpoint and add 'resnet.' prefix to all keys
checkpoint = torch.load("model_aider.pth", map_location=device)
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = "resnet." + k
    new_state_dict[new_key] = v

# Remove fc layer weights from checkpoint to avoid size mismatch
new_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith('resnet.fc.')}

model.load_state_dict(new_state_dict, strict=False)  # strict=False allows missing keys

model.to(device)
model.eval()

image_path = r"D:\Kranthi\damage_asses-main\mini_xbd\download.jpeg"
prediction = predict_image(model, image_path, class_names, device)
print(f"Prediction: {prediction}")
