import torch
import sys
import os

# Allow Python to find model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ResNet9


MODEL_PATH = "model/FingurePrintTOBloodGroup.pth"

device = torch.device("cpu")

print("Loading model...")

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

print("Model loaded successfully!")
print("Device:", device)
print("Classes:", 8)