import torch
from torchvision import transforms
from PIL import Image

# Maps incident class (predicted folder/class name) to damage level
incident_to_damage = {
    'collapsed_building': 'major-damage',
    'fire': 'major-damage',
    'flooded_areas': 'minor-damage',
    'traffic_incident': 'minor-damage',
    'normal': 'no-damage'
}

# Maps damage level to updated repair cost range
cost_mapping = {
    'no-damage': '₹0',
    'minor-damage': '₹500,000 to ₹2,000,000',
    'major-damage': '₹2,100,000 to ₹5,000,000',
    
}

def predict_image(model, image_path, class_names, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    pred_idx = predicted_class.item()
    incident_class = class_names[pred_idx]  # e.g., 'fire', 'flooded_areas'
    damage_class = incident_to_damage.get(incident_class, 'no-damage')
    repair_cost = cost_mapping.get(damage_class, 'Unknown')

    return incident_class, damage_class, repair_cost