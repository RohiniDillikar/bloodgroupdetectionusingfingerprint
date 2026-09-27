import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from predict import predict_image


image_path = "test_image.BMP"

result = predict_image(image_path)

print("\nPrediction Result")
print("-----------------")
print("Blood Group:", result["blood_group"])
print("Confidence:", result["confidence"], "%")