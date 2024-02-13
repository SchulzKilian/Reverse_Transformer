import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ai_hu_probabilities_test import get_data

class FloatsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(50, 100)  # Input layer with 50 features, outputting to hidden layer with 100 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)  # Another hidden layer
        self.fc3 = nn.Linear(50, 1)  # Output layer, outputting a single value
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid to ensure output is between 0 and 1
        return x

hu0,ai1 = get_data()
# Assuming hu0 and ai1 are already defined
assert len(hu0)== len(ai1)


X0 = torch.tensor(hu0[:320], dtype=torch.float32)
y0 = torch.zeros(320, 1, dtype=torch.float32)  # Create a tensor of zeros (label 0) for list0

X1 = torch.tensor(ai1[:320], dtype=torch.float32)
y1 = torch.ones(320, 1, dtype=torch.float32)  # Create a tensor of ones (label 1) for list1

# Combine the tensors to form a single dataset
X = torch.cat((X0, X1), dim=0)
y = torch.cat((y0, y1), dim=0)

dataset = FloatsDataset(X, y)

# Create a DataLoader
batch_size = 10  # Adjust as needed
shuffle = True  # Shuffle the data for training
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
X0 = torch.tensor(hu0[320:], dtype=torch.float32)
y0 = torch.zeros(len(hu0[320:]), 1, dtype=torch.float32)  # Create a tensor of zeros (label 0) for list0

X1 = torch.tensor(ai1[320:], dtype=torch.float32)
y1 = torch.ones(len(ai1[320:]), 1, dtype=torch.float32)  # Create a tensor of ones (label 1) for list1

# Combine the tensors to form a single dataset
X = torch.cat((X0, X1), dim=0)
y = torch.cat((y0, y1), dim=0)

dataset = FloatsDataset(X, y)

# Create a DataLoader
batch_size = 10  # Adjust as needed
shuffle = True  # Shuffle the data for training
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20  # Number of epochs for training

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for data, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')


from sklearn.metrics import precision_score

model.eval()  # Set the model to evaluation mode

# Lists to store actual and predicted labels
actuals = []
predictions = []

with torch.no_grad():  # No need to track gradients during evaluation
    for features, labels in test_loader:
        outputs = model(features)
        predicted_labels = (outputs > 0.5).float()  # Convert to binary labels (0 or 1)
        print(outputs)
        actuals.extend(labels.view(-1).tolist())
        predictions.extend(predicted_labels.view(-1).tolist())



precision = precision_score(actuals, predictions)
torch.save(model.state_dict(), 'simple_nn_model.pt')
print(f'Precision: {precision}')
