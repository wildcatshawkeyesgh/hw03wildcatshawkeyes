__all__ = ["binary_classification"]


import torch


def binary_cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-7  # to Prevent log(0) or log(1)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -torch.mean(
        y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    )
    return loss


def binary_classification(d, n, epochs=10000, eta=0.001):
    """
    Binary Classification with Linear and Nonlinear Layers
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = (X.sum(axis=1, keepdim=True) > 2).float()

    # W1 = torch.normal(mean = 0, std = torch.sqrt(torch.tensor(d)), size = (d, 48), requires_grad=True, dtype=torch.float32, device=device)

    current_dtype = torch.float32
    W1 = (
        torch.randn(d, 48, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / d, device=device, dtype=current_dtype))
    ).requires_grad_(True)
    W2 = (
        torch.randn(48, 16, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / 48, device=device, dtype=current_dtype))
    ).requires_grad_(True)
    W3 = (
        torch.randn(16, 32, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / 16, device=device, dtype=current_dtype))
    ).requires_grad_(True)
    W4 = (
        torch.randn(32, 1, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / 32, device=device, dtype=current_dtype))
    ).requires_grad_(True)

    train_losses = torch.zeros(epochs, device=device)

    W1_total = torch.zeros(epochs, d, 48, device=device)
    W2_total = torch.zeros(epochs, 48, 16, device=device)
    W3_total = torch.zeros(epochs, 16, 32, device=device)
    W4_total = torch.zeros(epochs, 32, 1, device=device)

    for epoch in range(epochs):
        Z1 = torch.matmul(X, W1)
        Z1 = torch.matmul(Z1, W2)
        A1 = 1 / (1 + torch.exp(-Z1))
        Z2 = torch.matmul(A1, W3)
        Z2 = torch.matmul(Z2, W4)
        A2 = 1 / (1 + torch.exp(-Z2))
        YPred = A2

        train_loss = binary_cross_entropy_loss(YPred, Y)

        train_loss.backward()

        with torch.no_grad():
            W1 -= eta * W1.grad
            W2 -= eta * W2.grad
            W3 -= eta * W3.grad
            W4 -= eta * W4.grad

            W1_total[epoch] = W1.clone()
            W2_total[epoch] = W2.clone()
            W3_total[epoch] = W3.clone()
            W4_total[epoch] = W4.clone()
            # Zero the gradients
            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()

            train_losses[epoch] = train_loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {train_loss.item():.4f}")

    return [train_losses, W1_total, W2_total, W3_total, W4_total]
