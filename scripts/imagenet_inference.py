from hw03wildcatshawkeyes import deepl

import sys
import getopt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
import matplotlib.pyplot as plt
import onnxruntime as ort
import polars as pl
from PIL import Image

import torch.optim as optim
from datetime import datetime
import torch.nn as nn
import os


file_path = "/data/CPE_487-587/imagenet-1k-arrow/"
dataset = load_from_disk(file_path)
class_names = dataset['train'].features['label'].names
test_dataset = dataset['test']
example = test_dataset[0]
image = example['image']
true_label = example['label']
true_name = class_names[true_label].split(',')[0].strip()
model_path = "image_model.onnx"
session = ort.InferenceSession(model_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = transform(image.convert('RGB')).unsqueeze(0).numpy()


input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor})
predicted_class = output[0].argmax()
predicted_name = class_names[predicted_class].split(',')[0].strip()

print(f"Predicted name: {predicted_name}")
print(f"True name: {true_name}")

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Predicted: {predicted_name}\nTrue: {true_name}")
plt.axis('off')
plt.savefig("prediction_result.png")
plt.close()
print("Saved image: prediction_result.png")