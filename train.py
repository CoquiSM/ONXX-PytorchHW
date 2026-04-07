import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv('diabetes.csv')

# X contains the 8 features, y contains the Outcome (0 or 1)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.int64)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tr = torch.from_numpy(X_train)
y_train_tr = torch.from_numpy(y_train)


x_means = X_train_tr.mean(0, keepdim=True)
x_deviations = X_train_tr.std(0, keepdim=True) + 0.0001


# Deep Learning Network
class DL_Net(nn.Module):
    def __init__(self, x_means, x_deviations):
        super().__init__()
        self.x_means = x_means
        self.x_deviations = x_deviations

        self.linear1 = nn.Linear(8, 16)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(16, 8)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(8, 2)
        self.act3 = nn.Softmax(dim=1)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        y_pred = self.act3(x)
        return y_pred


# Initialize Model, Loss, and Optimizer
model = DL_Net(x_means, x_deviations)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the Model
print("Training the model...")
epochs = 500
for epoch in range(epochs):
    y_pred = model(X_train_tr)
    loss = loss_fn(y_pred, y_train_tr)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Training complete!")

# Export to ONNX
model.eval().float()

# Create a dummy input
dummy_input = torch.randn(1, 8, dtype=torch.float32)

onnx_filename = "diabetes_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=["input1"],
    output_names=["output1"],
    opset_version=15
)
print(f"ONNX model saved as '{onnx_filename}'")