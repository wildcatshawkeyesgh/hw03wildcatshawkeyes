import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)


__all__ = ["SimpleNN", "ClassTrainer", "ConvAttention"]


class SimpleNN(nn.Module):
    def __init__(self, in_features, m):  # m is the number of classifications
        super(SimpleNN, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, m)  # check m vs m-1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConvAttention(nn.Module):
    def __init__(self, input_size=10, d_model=32, nhead=4, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.sig(x)
        return x
    
class ClassTrainer:
    def __init__(
        self,
        train_loader,
        test_loader,
        eta,
        epoch,
        loss,
        optimizer,
        model,
        device,
        scheduler,
        patience,
    ):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.eta = eta
        self.epoch = epoch
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.loss_vector = torch.zeros(epoch)
        self.accuracy_vector = torch.zeros(epoch)
        self.scheduler = scheduler
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def train(self):

        for i in range(self.epoch):
            epoch_loss = 0
            correct = 0
            total = 0
            for batch_features, batch_labels in self.train_loader:
                self.optimizer.zero_grad()
                predictions = self.model.forward(batch_features)
                loss = self.loss(predictions, batch_labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                predicted_classes = torch.argmax(predictions, dim=1)
                correct += (predicted_classes == batch_labels).sum().item()
                total += batch_labels.size(0)

            self.loss_vector[i] = epoch_loss / len(self.train_loader)
            self.accuracy_vector[i] = correct / total
            self.scheduler.step()

            if i == 0 or self.loss_vector[i] < self.best_loss:
                self.best_loss = self.loss_vector[i]
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping at epoch {i}")
                    break
            print(
                f"Epoch {i}, Loss: {self.loss_vector[i]:.4f}, Accuracy: {self.accuracy_vector[i]:.4f}"
            )

    def test(self):
        test_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in self.test_loader:
                test_outputs = self.model(batch_features)
                test_loss += self.loss(test_outputs, batch_labels).item()
        return test_loss / len(self.test_loader)

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(X)
            predicted_classes = torch.argmax(y_pred, dim=1)
            return predicted_classes

    def save(self, file_name=None):
        if file_name is None:
            file_name = "model.onnx"
        dummy_input = torch.zeros(1, self.model.in_features)
        torch.onnx.export(self.model, dummy_input, file_name)

    def evaluation(self):

        train_preds = []
        train_labels = []
        with torch.no_grad():
            for features, labels in self.train_loader:
                preds = torch.argmax(self.model(features), dim=1)
                train_preds.append(preds)
                train_labels.append(labels)
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)

        test_preds = []
        test_labels = []
        with torch.no_grad():
            for features, labels in self.test_loader:
                preds = torch.argmax(self.model(features), dim=1)
                test_preds.append(preds)
                test_labels.append(labels)
        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)

        plt.figure()
        plt.plot(self.loss_vector.numpy())
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("loss_curve.png")
        plt.show()

        # Plot accuracy
        plt.figure()
        plt.plot(self.accuracy_vector.numpy())
        plt.title("Accuracy over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("accuracy_curve.png")
        plt.show()

        # Confusion matrix
        plt.figure()
        cm = confusion_matrix(test_labels.numpy(), test_preds.numpy())
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        plt.show()

        # Training metrics
        train_accuracy = accuracy_score(train_labels.numpy(), train_preds.numpy())
        train_precision = precision_score(
            train_labels.numpy(), train_preds.numpy(), average="weighted"
        )
        train_recall = recall_score(
            train_labels.numpy(), train_preds.numpy(), average="weighted"
        )
        train_f1 = f1_score(
            train_labels.numpy(), train_preds.numpy(), average="weighted"
        )

        # Test metrics
        test_accuracy = accuracy_score(test_labels.numpy(), test_preds.numpy())
        test_precision = precision_score(
            test_labels.numpy(), test_preds.numpy(), average="weighted"
        )
        test_recall = recall_score(
            test_labels.numpy(), test_preds.numpy(), average="weighted"
        )
        test_f1 = f1_score(test_labels.numpy(), test_preds.numpy(), average="weighted")

        return (
            train_accuracy,
            train_precision,
            train_recall,
            train_f1,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
        )
