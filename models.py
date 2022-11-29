import torch
import torch.nn as nn
import numpy as np
from constants import LATENT_DIM, NUMBER_OF_PITCHES, BASIC_LENGTH


class TemporalVectors(nn.Module):
    def __init__(self, latent_vector_size, hidden_size, num_layers, sequence_length, device):
        super(TemporalVectors, self).__init__()
        self.latent_vector_size = latent_vector_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(latent_vector_size, hidden_size, num_layers, batch_first=True)

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, noise):
        latent_vectors = []

        out, _ = self.rnn(noise)
        latent_vectors.append(out)
        for _ in range(self.sequence_length - 1):
            out, _ = self.rnn(out)
            latent_vectors.append(out)

        return latent_vectors


class SequenceBarGenerator(nn.Module):
    def __init__(self, vectors_generator, bar_generator):
        super(SequenceBarGenerator, self).__init__()
        self.vectors_generator = vectors_generator
        self.bar_generator = bar_generator

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, noise):
        latent_vectors = self.vectors_generator(noise)

        latent_vectors = [v.reshape((-1, LATENT_DIM, 1, 1)) for v in latent_vectors]
        bars = [self.bar_generator(v) for v in latent_vectors]
        return torch.cat((bars), dim=2)


class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.t_conv = torch.nn.ConvTranspose2d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm2d(out_dim)
    
    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, x):
        x = self.t_conv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            GeneratorBlock(LATENT_DIM, 256, (4, 1), (4, 1)),
            GeneratorBlock(256, 128, (4, 1), (4, 1)),
            GeneratorBlock(128, 64, (1, 4), (1, 4)),
            GeneratorBlock(64, 32, (1, 3), (1, 1)),
            GeneratorBlock(32, 16, (4, 1), (4, 1)),
            GeneratorBlock(16, 1, (1, 12), (1, 12))
        )

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, x):
        x = x.reshape(-1, LATENT_DIM, 1, 1)
        x = self.main(x)
        x = x.view(-1, 1, BASIC_LENGTH, NUMBER_OF_PITCHES)
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, x):
        x = self.conv(x)
        x = self.layernorm(x)
        return torch.nn.functional.leaky_relu(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sequence_multiplier = 3
        self.main = nn.Sequential(
            DiscriminatorBlock(1, 16, (1, 12), (1, 12)),
            DiscriminatorBlock(16, 16, (4 * sequence_multiplier, 1), (4 * sequence_multiplier, 1)),
            DiscriminatorBlock(16, 64, (1, 3), (1, 1)),
            DiscriminatorBlock(64, 64, (1, 4), (1, 4)),
            DiscriminatorBlock(64, 128, (4, 1), (4, 1)),
            DiscriminatorBlock(128, 128, (2, 1), (1, 1)),
            DiscriminatorBlock(128, 256, (3, 1), (3, 1))
        )
        self.dense = torch.nn.Linear(256, 1)

    def __str__(self) -> str:
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward(self, x):
        x = self.main(x)
        x = x.reshape(-1, 256)
        x = self.dense(x)
        return x