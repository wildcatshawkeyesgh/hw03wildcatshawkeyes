from hw03wildcatshawkeyes import deepl

from segmentation_models_pytorch.losses import DiceLoss
from monai.losses import TverskyLoss

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
eta = 0.001
epoch = 200
patience = 7
batch_size = 1024

processor = deepl.DataProcessor(input_folder=file_location, output_folder=".")
processor.process_all()

data = deepl.DataPrep(data_path="./Final.csv", batch_size=batch_size)
data.dataload()
data = deepl.DataPrep(data_path="./Final.csv", batch_size=batch_size)
data.dataload()

# ============ ADD THIS SECTION ============
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Get the raw data before it was batched
features, results = data.get_features_results()
X = features.to_numpy()
y = results.to_numpy().squeeze()

# Quick split (or reuse your existing split)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit logistic regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

print("\n===== LOGISTIC REGRESSION BASELINE =====")
print(f"Baseline (majority class): {max(y_train.mean(), 1 - y_train.mean()):.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(classification_report(y_test, y_pred))

print("\n===== FEATURE IMPORTANCE BY LAG =====")
feature_names = ["speed"] + [f"speed_{i}" for i in range(1, 10)]
for name, coef in zip(feature_names, lr_model.coef_[0]):
    print(f"{name}: {coef:.4f}")
print("========================================\n")
# ============ END SECTION ============

model = deepl.OptimusPrime()

model = deepl.OptimusPrime()

# loss = nn.BCELoss()
loss = DiceLoss(mode="binary", from_logits=True)
# loss = TverskyLoss(sigmoid=False, alpha=0.3, beta=0.7)
optimizer = optim.AdamW(model.parameters(), lr=eta, weight_decay=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

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
