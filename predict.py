# import torch
# from torchvision import transforms
# from PIL import Image
# import os

# def predict_image(model, image_path, class_names, device):
#     # Preprocessing the image
#     image = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
#     image = transform(image).unsqueeze(0).to(device)

#     # Predict the class
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         _, predicted_class = torch.max(output, 1)
    
#     # Map prediction to class name
#     predicted_label = class_names[predicted_class.item()]
    
#     # For the test, return both prediction and the ground truth label
#     # Here, assuming ground truth is available in a label file or as part of the dataset
#     return predicted_label  # For testing, you may need to get the true label from your data
import torch
from torchvision import transforms
from PIL import Image
from model import DamageClassifier  # make sure your model class is imported

# Prediction function
def predict_image(model, image_path, class_names, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    pred_idx = predicted_class.item()
    if pred_idx >= len(class_names):
        return "Unknown class"
    return class_names[pred_idx]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']

    model = DamageClassifier()
    model.load_state_dict(torch.load("model_aider.pth", map_location=device))
    model.to(device)

    test_image_path = "test_images/sample.jpg"  # change to your test image path
    prediction = predict_image(model, test_image_path, class_names, device)
    print(f"Predicted damage class: {prediction}")
