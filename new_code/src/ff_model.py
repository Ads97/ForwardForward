import math

import torch
import torch.nn as nn

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers
        self.act_fn = ReLU_full_grad()

        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(784, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers)
        ]

        # [784,2000,2000,2000]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        ) # 2000+2000+2000 = 6000
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        ) # 6000, 10
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    # loss incentivizing the mean activity of neurons in a layer to have low variance
    def _calc_peer_normalization_loss(self, idx, z): # z is bs*2, 2000
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0) #bsx2000 -> 2000

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        ) # the detach means that the gradient because of previous batches is not backpropagated. only the current mean activity is backpropagated
        # running_mean * 0.9 + mean_activity * 0.1

        # 2000
        # 1 = mean activation across entire layer
        # 

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1) # sum of squares of each activation. bs*2

        # print("sum of squares shape: ", sum_of_squares.shape)
        # exit()
        # s - thresh    --> sigmoid --> cross entropy

        logits = sum_of_squares - z.shape[1] # if the average value of each activation is >1, logit is +ve, else -ve.
        ff_loss = self.ff_loss(logits, labels.float()) # labels are 0 or 1, so convert to float. logits->sigmoid->normal cross entropy

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels) # threshold is logits=0, so sum of squares = 784 
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # print(inputs["pos_images"].shape) # bs, 1, 28, 28
        # print(inputs["neg_images"].shape) # bs, 1, 28, 28
        # print(inputs["neutral_sample"].shape) # bs, 1, 28, 28
        # print(labels["class_labels"].shape) # bs
        # exit()
        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0) # 2*bs, 1, 28, 28
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device) # 2*bs
        posneg_labels[: self.opt.input.batch_size] = 1 # first BS samples true, next BS samples false

        z = z.reshape(z.shape[0], -1) # 2*bs, 784
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z) 

        input_classification_model = []

        # 784, 2000, 2000, 2000

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    # print(z.shape)
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1) # concat all activations from all layers

        # print(input_classification_model.shape)
        # exit()

        # [0.5, 1, 1.5, ....]
        # max = 3
        # [-2.5, -2, -1.5, .. 0, ..]

        output = self.linear_classifier(input_classification_model.detach()) # bs x 10 , 
        output = output - torch.max(output, dim=-1, keepdim=True)[0] # not entirely clear why each entry in output is made 0 or -ve
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs

# unclear as to why normal relu doesn't work
class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
