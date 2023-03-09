import torch.nn as nn
import torch

from modules.helpers.transformer import Transformer


class AutoencoderV0(nn.Module):
    def __init__(self, args):
        super(AutoencoderV0, self).__init__()
        self.args = args

        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_basic = nn.Linear(args.emb, 6)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        q_enemies_list = []

        # each enemy has an output Q
        for i in range(self.args.enemy_num):
            q_enemy = self.q_basic(outputs[:, 1 + i, :])
            q_enemy_mean = torch.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        return q, h
