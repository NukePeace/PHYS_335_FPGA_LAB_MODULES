import torch

# Define a model
class MyModel(torch.nn.Module):
    def __init__(self, input_size, output_size): # INIT W input_size=10, output_size=5
        super(MyModel, self).__init__()
        self.input_layer = torch.nn.Linear(10, 10)
        self.hidden_layer1 = torch.nn.Linear(10, 32)
        self.output_layer2 = torch.nn.Linear(32, 16)
        self.output_layer3 = torch.nn.Linear(16, 5)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.sigmoid(self.hidden_layer1(x))
        x = torch.sigmoid(self.output_layer2(x)) # dropout 30% after this layer
        x = torch.dropout(x, p=0.3, training=self.training)
        x = torch.softmax(self.output_layer3(x))
        return x