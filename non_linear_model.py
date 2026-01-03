import torch
import torch.nn as nn
import torch.optim as optim

import helper_utils

# Combined dataset for bikes and cars
distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times for dataset.
times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]], dtype=torch.float32)


"""
Normalize the data.
"""

# Calculate mean and standard deviation for 'distance' tensor.
distances_mean = distances.mean()
distances_std = distances.std()

# Calculate mean and standard deviation  for 'times' tensor.
times_mean = times.mean()
times_std = times.std()

# Apply standardization to the distances & times.
distances_normalized = (distances - distances_mean) / distances_std
times_normalized = (times - times_mean) / times_std

"""
Create our model
"""

# Seed for torch
torch.manual_seed(27)

model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

"""
Train the model
"""

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3000):
    optimizer.zero_grad()  # Reset every loop. We do not want it to start from where it left off.
    outputs = model(distances_normalized) # Make predictions (forward pass)
    loss = loss_function(outputs, times_normalized) # Calculate the loss
    loss.backward() # Calculate adjustments (backward pass)
    optimizer.step() # Update model's parameters

    # Create live plots every 50 epochs (loops)
    if (epoch + 1) % 50 == 0:
        helper_utils.plot_training_progress(
            epoch = epoch,
            loss = loss,
            model = model,
            distances_norm = distances_normalized,
            times_norm =  times_normalized
        )
print("\nTraining Completed.")
print(f"\nFinal Loss: {loss.item()}")

helper_utils.plot_final_fit(model, distances, times, distances_normalized, times_std, times_mean)