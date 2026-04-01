from hw03wildcatshawkeyes import deepl

import sys
import getopt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
import matplotlib.pyplot as plt

import polars as pl

import torch.optim as optim
from datetime import datetime
import torch.nn as nn
import os

__all__ = ["ImageCNN", "ClassTrainer", "OptimusPrime","CNNTrainer"]

file_path = "/data/CPE_487-587/imagenet-1k-arrow/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


device_id = deepl.get_best_gpu(strategy="utilization")
device = torch.device(f"cuda:{device_id}")
print(f"Selected GPU: {device_id}")

epochs = 10000
train_ratio = 0.01
val_ratio = 0.01
test_ratio = 0.01
#hooks for shell script
opts, args = getopt.getopt(
    sys.argv[1:],
    "",
    ["epochs=", "train_ratio=", "val_ratio=", "test_ratio="],
)
for opt, val in opts:
    if opt == "--epochs":
        epochs = int(val)
    elif opt == "--train_ratio":
        train_ratio = float(val)
    elif opt == "--val_ratio":
        val_ratio = float(val)
    elif opt == "--test_ratio":
        test_ratio = float(val)

#loads data
dataset = load_from_disk(file_path)

train_dataset = dataset['train']
val_dataset = dataset['validation']


num_classes = len(train_dataset.features['label'].names)
class_names = train_dataset.features['label'].names
print(f"Number of classes: {num_classes}")

train_size = int(len(train_dataset) * train_ratio)
val_size = int(len(val_dataset) * val_ratio)
#test_size = int(len(test_dataset) * test_ratio)

train_dataset = train_dataset.select(range(train_size))
val_dataset = val_dataset.select(range(val_size))
#test_dataset = test_dataset.select(range(test_size))


#save ex images
script_dir = os.path.dirname(os.path.abspath(__file__))
train_example = train_dataset[0]
train_image = train_example['image']
train_label_id = train_example['label']
train_label_name = class_names[train_label_id].split(',')[0].strip()

plt.figure(figsize=(8, 8))
plt.imshow(train_image)
plt.title(f"Training Example\nID {train_label_id}: {train_label_name}")
plt.axis('off')
plt.savefig(os.path.join(script_dir, "example_train_image.png"), bbox_inches='tight')
plt.close()

val_example = val_dataset[0]
val_image = val_example['image']
val_label_id = val_example['label']
val_label_name = class_names[val_label_id].split(',')[0].strip()

plt.figure(figsize=(8, 8))
plt.imshow(val_image)
plt.title(f"Validation Example\nID {val_label_id}: {val_label_name}")
plt.axis('off')
plt.savefig(os.path.join(script_dir, "example_val_image.png"), bbox_inches='tight')
plt.close()

def preprocess_train(examples):
    images = [train_transform(img.convert('RGB')) for img in examples['image']]
    labels = examples['label']
    return {'pixel_values': images, 'labels': labels}

def preprocess_val(examples):
    images = [val_transform(img.convert('RGB')) for img in examples['image']]
    labels = examples['label']
    return {'pixel_values': images, 'labels': labels}

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = train_dataset.with_transform(preprocess_train)
val_dataset = val_dataset.with_transform(preprocess_val)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    pin_memory=True,
    collate_fn=collate_fn,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
    num_workers=4
)
print(f"Train samples: {len(train_dataset)}")
print(f"Train batches: {len(train_loader)}")

eta = 0.01
loss = nn.CrossEntropyLoss()
model = deepl.ImageCNN(num_classes=num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=eta, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma=1)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

trainer = deepl.CNNTrainer(
    train_loader=train_loader,
    test_loader=val_loader,
    eta=eta,
    epoch=epochs,
    loss=loss,
    optimizer=optimizer,
    model=model,
    device=device,
    scheduler=scheduler,
    
)
trainer.train()
trainer.save("image_model.onnx")

train_acc, train_prec, train_rec, train_f1, test_acc, test_prec, test_rec, test_f1 = (
    trainer.evaluation()
)


df_metrics = pl.DataFrame(
    {
        "metric": ["accuracy", "precision", "recall", "f1"],
        "train": [train_acc, train_prec, train_rec, train_f1],
        "test": [test_acc, test_prec, test_rec, test_f1],
    }
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "hw03_metrics.csv"

df_metrics = pl.DataFrame(
    {
        "timestamp": [timestamp],
        "eta": [eta],
        "epoch": [epoch],
        "train_accuracy": [train_acc],
        "train_precision": [train_prec],
        "train_recall": [train_rec],
        "train_f1": [train_f1],
        "test_accuracy": [test_acc],
        "test_precision": [test_prec],
        "test_recall": [test_rec],
        "test_f1": [test_f1],
    }
)

if os.path.exists(filename):
    existing = pl.read_csv(filename)
    df_metrics = pl.concat([existing, df_metrics])

df_metrics.write_csv(filename)
