import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_positions=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_positions * 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Clamp output
        x = torch.clamp(x, -1, 1)

        return x