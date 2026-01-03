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

# Train the model
for epoch in range(50000):
    # Reset the optimizer's gradients'
    optimizer.zero_grad()
    # Make predictions
    outputs = model(distances)
    # Calculate the loss / difference
    loss = loss_function(outputs, times)
    # Calculate adjustments
    loss.backward()
    # Update the model's parameter
    optimizer.step()
    # Print loss every 50 epochs
    if (epoch + 1) % 50  == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

helper_utils.plot_results(model, distances, times)

distance_to_predict =7.0

# torch.no_grad() for more efficient predictions
with torch.no_grad():
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)

    predicted_time = model(new_distance)

    print(f"Prediction for a {distance_to_predict}- mile delivery: {predicted_time.item():.1f} minutes")

    if predicted_time > 30:
        print("\nMore than 30 minutes")
    else:
        print("\nLess than 30 minutes")

# View our model's parameters it found  was best
layer = model[0]

weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weight: {weights}")
print(f"Bias: {bias}")