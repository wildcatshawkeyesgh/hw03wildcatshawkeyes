from hw03wildcatshawkeyes import deepl

from segmentation_models_pytorch.losses import TverskyLoss

import polars as pl
import torch
import torch.optim as optim
from datetime import datetime
import torch.nn as nn
import os


file_location = "/data/CPE_487-587/ACCDataset/"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

device_id = deepl.get_best_gpu(strategy="utilization")
device = torch.device(f"cuda:{device_id}")
print(f"Selected GPU: {device_id}")
eta = 0.0009
epoch = 6
patience = 25
batch_size = 4096
num_lags = 9
processor = deepl.DataProcessor(input_folder=file_location, output_folder=".", num_lags = num_lags)
processor.process_all()

data = deepl.DataPrep(
    data_path="./Final.csv",
    batch_size=batch_size)
data.dataload()


model = deepl.OptimusPrime().to(device)


# loss = nn.BCELoss()
# loss = DiceLoss(mode="binary", from_logits=True)
loss = TverskyLoss(mode="binary", alpha=0.7, beta=0.3, gamma = 2, from_logits=True)
optimizer = optim.AdamW(model.parameters(), lr=eta, weight_decay=0.00001)
warm = optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, total_iters = 10)
learn_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warm, learn_rate], milestones=[10])

trainer = deepl.ClassTrainer(
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
