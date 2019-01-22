import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Unity Network
"""

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, duel=False):
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

        self.fc1 = nn.Linear(state_size, 128)

        self.fc2 = nn.Linear(128, 128) #-> common_out

        # duel: output action values
        self.fc3a = nn.Linear(128, 8)

        self.fc4a = nn.Linear(8, action_size)

        ####################################

        # duel: output advantage value
        self.fc3v = nn.Linear(128, 4)

        self.fc4v = nn.Linear(4, 1)

        ####################################

        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state_inputs):
        """Build a network that maps state -> action values."""

        # common: one linear relu layer
        x = F.relu(self.fc1(state_inputs))

        # common: one linear relu layer for action
        common_out = F.relu(self.fc2(x))

        # if duel network is applied
        if self.duel:
            # for actions
            a = F.relu(self.fc3a(common_out))
            a = self.fc4a(a)

            # for values
            v = F.relu(self.fc3v(common_out))
            v = self.fc4v(v)

            #A(sa') - 1/|A|*A(sa')p
            a_adj = a - a.mean(dim=1, keepdim=True)

            out = v + a_adj
        else:
            # one linear output layer for actions
            out = self.fc3(common_out)

        # final output
        return out
