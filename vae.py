import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Independent, Normal
from torch.nn import functional as F


class ProductOfCategoricals(Distribution):
    def __init__(self, logits, num_classes):
        self.categoricals = [Categorical(logits=l) for l in logits.split(tuple(num_classes), dim=1)]

    def log_prob(self, value):
        return torch.cat([categorical.log_prob(v) for v, categorical in zip(value.chunk(value.size(1), dim=1), self.categoricals)], dim=1).sum(dim=1)

    def sample(self, sample_shape=torch.Size([])):
        return torch.stack([categorical.sample(sample_shape) for categorical in self.categoricals], dim=1)


class FCEncoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * latent_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)


class FCDecoder(nn.Module):
    def __init__(self, output_size, latent_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        return self.fc2(h1)


class EmbeddingEncoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(num_embeddings, hidden_size // len(input_size)) for num_embeddings in input_size)
        self.fc2 = nn.Linear(len(input_size) * (hidden_size // len(input_size)), 2 * latent_size)

    def forward(self, x):
        h1 = F.relu(torch.cat([embed(x_i.squeeze(dim=1)) for embed, x_i in zip(self.embeddings, x.chunk(x.size(1), dim=1))], dim=1))
        return self.fc2(h1)


class CategoricalDecoder(nn.Module):
    def __init__(self, output_size, latent_size, hidden_size):
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, torch.sum(output_size).item())

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        return self.fc2(h1)


class VAE(nn.Module):
    def __init__(self, input_shape, latent_size, hidden_size, observation):
        super().__init__()
        self.input_shape, self.latent_size, self.observation = input_shape, latent_size, observation
        if len(input_shape) == 1:
            self.encoder = FCEncoder(input_shape[0], latent_size, hidden_size)
            self.decoder = FCDecoder(2 * input_shape[0], latent_size, hidden_size)
        else:
            self.encoder = EmbeddingEncoder(input_shape, latent_size, hidden_size)
            self.decoder = CategoricalDecoder(input_shape, latent_size, hidden_size)

    def encode(self, x):
        posterior_params = self.encoder(x)
        return Independent(Normal(posterior_params[:, :self.latent_size], posterior_params[:, self.latent_size:].exp()), 1)

    def decode(self, z):
        obs_params = self.decoder(z)
        if self.observation == 'gaussian':
            return Independent(Normal(obs_params[:, :self.input_shape[0]], obs_params[:, self.input_shape[0]:].exp()), len(self.input_shape))
        elif self.observation == 'categorical':
            return ProductOfCategoricals(logits=obs_params, num_classes=self.input_shape)

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.rsample()
        return self.decode(z), posterior, z
