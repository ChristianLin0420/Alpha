import torch
import torch.nn as nn

class LinearLayerAutoEncoder(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.input_dim = args.max_input_dim
        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.input_dim)
        )

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, x):

        latent = self.encoder(x)
        decoded = self.decoder(latent)

        reconstruction_loss = self.mse(x, decoded) * self.args.reconstruction_lambda
        reconstruction_loss = torch.mean(reconstruction_loss, -1)

        return latent, reconstruction_loss.detach()