import torch.nn as nn
import torch

from modules.helpers.auto_encoder import LinearLayerAutoEncoder


class AutoencoderV0(nn.Module):
    def __init__(self, input_shape, args):
        super(AutoencoderV0, self).__init__()
        self.args = args

        self.autoencoder = LinearLayerAutoEncoder(args)
        self.hidden_net = nn.GRUCell(32, args.hidden_size)
        self.policy_net = nn.Linear(32 + args.hidden_size, args.max_action_size)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.hidden_size).to(self.args.device)

    def forward(self, inputs, hidden_state):
        inputs = inputs.view(-1, self.args.max_input_dim)
        hidden_state = hidden_state.view(-1, self.args.hidden_size)

        latent_inputs, reconstruction_loss = self.autoencoder(inputs)
        hidden_state = self.hidden_net(latent_inputs, hidden_state)

        policy_inputs = torch.cat([latent_inputs, hidden_state], dim = -1)

        q = self.policy_net(policy_inputs)
        
        return q, hidden_state, reconstruction_loss
