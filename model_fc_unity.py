import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Unity Network
"""

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=0, duel=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.duel = duel

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64) #-> common_out

        # duel: output action values
        self.fc4a = nn.Linear(64, 64)

        self.fc5a = nn.Linear(64, action_size)

        ####################################

        # duel: output advantage value
        self.fc4v = nn.Linear(64, 64)

        self.fc5v = nn.Linear(64, 1)

        ####################################

        self.fc6 = nn.Linear(64, action_size)

    def forward(self, state_inputs, actions=None):
        """Build a network that maps state -> action values."""

        # common: one linear relu layer
        x = F.relu(self.fc1(state_inputs))

        # common: one linear relu layer
        x = F.relu(self.fc2(x))

        # common: one linear relu layer for action
        common_out = F.relu(self.fc3(x))

        # if duel network is applied
        if self.duel:
            # for actions
            a = F.relu(self.fc4a(common_out))
            a = self.fc5a(a)

            # for values
            v = F.relu(self.fc4v(common_out))
            v = self.fc5v(v)

            #A(sa') - 1/|A|*A(sa')p
            a_adj = a - a.mean(dim=1, keepdim=True)

            out = v + a_adj
        else:
            # one linear output layer for actions
            out = self.fc6(common_out)

        # final output
        return out
