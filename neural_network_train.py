import torch
import torch.nn as nn
import torch.optim as optim

class TextClassifier(nn.Module):
    def __init__(self, input_dimension):
        super(TextClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)



from torch.utils.data import TensorDataset, DataLoader

batch_size = 32
epochs = 10
X_train = ""
Y_train =""
# terning datasets to tensors
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# start model
input_dimension = X_train.shape[1]  # fit shape
model = TextClassifier(input_dimension)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs.squeeze(), labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}')
    