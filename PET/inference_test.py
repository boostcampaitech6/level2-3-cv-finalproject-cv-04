import torch
from torch.autograd import profiler

# Define your model
class YourModel(torch.nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define your layers here
        self.linear1 = torch.nn.Linear(224, 224)
        self.linear2 = torch.nn.Linear(224, 224)
        self.linear3 = torch.nn.Linear(224, 224)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # Define the forward pass of your model
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        return x

# Instantiate your model and move it to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModel()

# Input data (example) and move it to the GPU
input_data = torch.randn(1, 3, 224, 224)  # Adjust the shape as per your input size

# Use profiler to profile the model's inference on the GPU
with profiler.profile(record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        # Execute the forward pass on the GPU
        model(input_data)

# Print the profiling results
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))