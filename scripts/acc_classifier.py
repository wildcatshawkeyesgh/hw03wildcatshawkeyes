from hw03wildcatshawkeyes import DataProcessor, DataPrep, ClassTrainer, ConvAttention
from segmentation_models_pytorch.losses import DiceLoss

import polars as pl
import torch
import torch.optim as optim
from datetime import datetime

import os


file_location = ""
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)


eta = 0.01
epoch = 200
patience = 7
batch_size = 1024

processor = DataProcessor(input_folder=file_location, output_folder=".")
processor.process_all()

data = DataPrep(data_path="./Final.csv", batch_size=batch_size)
data.dataload()


model = ConvAttention()

loss = DiceLoss(mode="binary")
optimizer = optim.Adam(model.parameters(), lr=eta)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

trainer = ClassTrainer(
    train_loader=data.train_loader,
    test_loader=data.test_loader,
    eta=eta,
    epoch=epoch,
    loss=loss,
    optimizer=optimizer,
    model=model,
    device=device,
    scheduler=scheduler,
    patience=patience,
)
trainer.train()
trainer.test()
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
