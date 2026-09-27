import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import ResNet9


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "FingurePrintTOBloodGroup.pth"
)

CLASS_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "class_mapping.json"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")


# --------------------------------------------------
# Load class mapping
# --------------------------------------------------

with open(CLASS_MAPPING_PATH, "r") as f:
    mapping = json.load(f)

idx_to_class = {
    int(k): v
    for k, v in mapping["idx_to_class"].items()
}


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = ResNet9(
    in_channels=3,
    num_classes=8
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_path):

    # Open image and convert RGBA/RGB/etc. → RGB
    image = Image.open(image_path).convert("RGB")

    # Apply same preprocessing used during validation/test
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    predicted_class = idx_to_class[predicted_idx]

    return {
        "blood_group": predicted_class,
        "confidence": round(confidence * 100, 2)
    }