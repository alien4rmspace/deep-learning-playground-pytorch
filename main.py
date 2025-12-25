import torch
import torch.nn as nn
import torch.optim as optim

import helper_utils

torch.manual_seed(42)

# Distance in miles
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

# Corresponding delivery times
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# Define the model
model = nn.Sequential(nn.Linear(1,1))

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)