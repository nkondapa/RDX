import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(MLP, self).__init__()

        assert num_layers >= 2, 'Number of layers must be at least 2'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.out_layer = torch.nn.Linear(hidden_size, output_size)

        self.hidden_layers = []
        if num_layers > 2:
            self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])

        self.layers = [self.input_layer] + [l for l in self.hidden_layers] + [self.out_layer]
        self.relu = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(0.5)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.dropout(out)

        for layer in self.hidden_layers:
            out = layer(out)
            out = self.relu(out)

        out = self.out_layer(out)

        return out

