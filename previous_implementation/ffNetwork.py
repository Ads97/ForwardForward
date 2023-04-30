import torch
import torch.nn as nn
from torch.optim import Adam
from utilities import *
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, config, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=config['lr'])
        self.threshold = config['threshold']
        self.num_epochs = config['epochs']

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4) #normalise the input to prevent the magnitude from affecting the layer computation
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        act_pos = self.forward(x_pos)
        g_pos = act_pos.pow(2).mean(1)
        act_neg = self.forward(x_neg)
        g_neg = act_neg.pow(2).mean(1)
        loss = torch.log(1+torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return act_pos.detach(), act_neg.detach(),loss.item()

class Net(torch.nn.Module):
    def __init__(self, dims, config):

        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers = self.layers + [Layer(dims[d], dims[d + 1], config).to(DEVICE)]

    def predict(self, x,num_classes=10):
        goodness_per_label = []
        for label in range(num_classes):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer.forward(h)
                goodness = goodness + [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, dim=1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        loss=0.0
        for i, layer in enumerate(self.layers):
            # print("training layer: ", i)
            h_pos, h_neg,layer_loss = layer.train(h_pos, h_neg)
            loss+=layer_loss
        return loss / len(self.layers)