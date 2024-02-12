import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence  # Import for padding sequences
from ai_hu_probabilities_test import get_data

hu0, ai1 = get_data()
import torch

# Assuming hu0 and ai1 are lists of lists representing sequences
# Convert each inner list to a tensor
hu0_tensors = [torch.tensor(seq) for seq in hu0]
ai1_tensors = [torch.tensor(seq) for seq in ai1]

# Now hu0_tensors and ai1_tensors contain tensors representing sequences
# Pad sequences to the same length using pad_sequence
padded_hu0 = torch.nn.utils.rnn.pad_sequence(hu0_tensors, batch_first=True, padding_value=0)
padded_ai1 = torch.nn.utils.rnn.pad_sequence(ai1_tensors, batch_first=True, padding_value=0)

# Concatenate the padded sequences along dimension 0
data = torch.cat((padded_hu0, padded_ai1), dim=0)




# Now padded_hu0 and padded_ai1 will have the same length
# You can concatenate them along dimension 0 to form the input data tensor
data = torch.cat((padded_hu0, padded_ai1), dim=0)

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

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # No need to convert to tensor here
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Convert labels to float32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example data
data = torch.cat((padded_ai1, padded_hu0), dim=0)  # Concatenate the padded sequences along dimension 0

# Assign single label for each list
labels = [0.0] * len(ai1) + [1.0] * len(hu0)

# Create dataset and DataLoader
dataset = CustomDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

input_dimension = max_length  # Assuming the maximum sequence length is the input dimension
model = TextClassifier(input_dimension)

# Step 2: Define your loss function
criterion = nn.BCELoss()

# Step 3: Define your optimizer
optimizer = optim.Adam(model.parameters())

# Step 4: Training loop
epochs = 10
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}')
