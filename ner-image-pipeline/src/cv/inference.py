import torch
from torchvision import transforms
from PIL import Image
from src.utils.device import get_device
from src.cv.train import build_model


def load_model(model_path="models/cv_model/resnet18.pth"):
    device = get_device()
    
    # Build model architecture
    model = build_model()

    # Load trained weights
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    # Move model to device
    model.to(device)
    # Switch to evaluation mode
    model.eval()

    return model


def predict(image_path, idx_to_class, model_path="models/cv_model/resnet18.pth"):
    # Select device for inference
    device = get_device()
    # Load trained model
    model = load_model(model_path)

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # Resize image to ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(  # ImageNet normalization
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    # Apply transforms and add batch dimension
    image = transform(image).unsqueeze(0).to(device)

    # Disable gradient calculation because we only need inference (evaluation)
    with torch.no_grad():

        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item() # Get predicted class index

    # Convert index to class name
    return idx_to_class[pred]