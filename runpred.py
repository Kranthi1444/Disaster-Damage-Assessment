import torch
from model import DamageClassifier
from predict import predict_image  # your updated prediction function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These are the incident type class labels used during training
class_names = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']

# Mapping incident class to damage level
incident_to_damage = {
    'collapsed_building': 'destroyed',
    'fire': 'major-damage',
    'flooded_areas': 'minor-damage',
    'traffic_incident': 'minor-damage',
    'normal': 'no-damage'
}

# Mapping damage level to estimated cost
cost_mapping = {
    'no-damage': 0,
    'minor-damage': 5000,
    'major-damage': 50000,
    'destroyed': 500000
}

# Load your trained model
model = DamageClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load("model_aider.pth", map_location=device))
model.to(device)
model.eval()

# Path to test image
image_path = r"D:\Kranthi\damage_asses-main\mini_xbd\download.jpeg"

# Predict incident class
incident_label = predict_image(model, image_path, class_names, device)[0]

# Map to damage class and cost
damage_class = incident_to_damage.get(incident_label, 'no-damage')
repair_cost = cost_mapping.get(damage_class, 0)

print(f"Incident Class: {incident_label}")
print(f"Mapped Damage Class: {damage_class}")
print(f"Estimated Repair Cost: ₹{repair_cost:,}")
