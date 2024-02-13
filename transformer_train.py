import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ai_hu_probabilities_test import get_data


class TransformerModel(nn.Module): # this transformer didnt yield sufficient results so we changed to the one in the old_testing fil
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src):
        embedded_src = self.embedding(src)
        src_mask = self.generate_square_subsequent_mask(src.size(1))
        output = self.transformer_encoder(embedded_src, src_mask)
        output = self.fc(output[:, -1, :])  # Take only the output of the last time step
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class FloatsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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



