import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from sumTree import SumTree

#from model_Atari_3D_duel import QNetwork
#from model_Atari_3D import QNetwork
from model_fc_unity import QNetwork

"""
This version is relatively more stable:
- TD error prioritized replay
- double and dual network
- TD error update and weight adjustment
- added error clipping
- used deque rotation instead of indexing for quicker update
- added memory index for quicker calculation
"""

BUFFER_SIZE = int(1e3)        # replay buffer size #int(1e6)
BATCH_SIZE = 64               # minibatch size ＃128
REPLAY_MIN_SIZE = int(1e3)    # min len of memory before replay start int(1e5)
GAMMA = 0.99                  # discount factor
TAU = 1e-3                    # for soft update of target parameters
LR = 1e-4                     # learning rate #1e-3
LR_DECAY = True               # decay learning rate?
LR_DECAY_START = int(4e5)     # number of steps before lr decay starts
LR_DECAY_STEP = int(5e3)      # LR decay steps
LR_DECAY_GAMMA = 0.999        # LR decay gamma
UPDATE_EVERY = 16             # how often to update the network #4
TD_ERROR_EPS = 1e-3           # make sure TD error is not zero
P_REPLAY_ALPHA = 0.6          # balance between prioritized and random sampling #0.7
P_REPLAY_BETA = 0.4           # adjustment on weight update #0.5
P_BETA_DELTA = 1e-6           # beta increment per sampling
USE_DUEL = False              # use duel network? V and A?
USE_DOUBLE = False            # use double network to select TD value?
REWARD_SCALE = False          # use reward clipping?
ERROR_CLIP = False            # clip error
ERROR_MAX = 1.0               # max value of error if do clipping
ERROR_INIT = False            # set an init value for error
USE_TREE = True               # use tree for memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # object reference to constant values:
        self.lr_decay = LR_DECAY
        self.p_replay_alpha = P_REPLAY_ALPHA
        self.p_replay_beta = P_REPLAY_BETA
        self.reward_scale = REWARD_SCALE
        self.error_clip = ERROR_CLIP

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, USE_DUEL).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, USE_DUEL).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   LR_DECAY_STEP,
                                                   gamma=LR_DECAY_GAMMA)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, TD_ERROR_EPS, seed,
                                   P_REPLAY_ALPHA, REWARD_SCALE, ERROR_CLIP,
                                   ERROR_MAX, ERROR_INIT, USE_TREE)

        # Initialize time step (for updating every UPDATE_EVERY steps and others)
        self.t_step = 0
        # keep track on whether training has started
        self.isTraining = False
        self.print_params()

    def print_params(self):
        print("current device: {}".format(device))
        print("use duel network (a and v): {}".format(USE_DUEL))
        print("use double network: {}".format(USE_DOUBLE))
        print("use reward scaling: {}".format(REWARD_SCALE))
        print("use error clipping: {}".format(ERROR_CLIP))
        print("buffer size: {}".format(BUFFER_SIZE))
        print("batch size: {}".format(BATCH_SIZE))
        print("initial learning rate: {}".format(LR))
        print("learing rate decay: {}".format(LR_DECAY))
        print("min replay size: {}".format(REPLAY_MIN_SIZE))
        print("target network update: {}".format(UPDATE_EVERY))
        print("optimizer: {}".format(self.optimizer))

    def get_TD_values(self, local_net, target_net, s, a, r, ns, d, isLearning=False):

        ###### TD TARGET #######
        s, ns, a = s.float().to(device), ns.float().to(device), a.to(device)
        with torch.no_grad(): #for sure no grad for this part

            ns_target_vals = target_net(ns)

            #0:the value, 1: argmax, unsqueeze to match the side of TD current
            if USE_DOUBLE:
                if np.random.rand() > 0.5:
                    # use target network to get value
                    ns_target_vals_tn = target_net(ns)
                    # use local network to get argmax
                    ns_target_max_arg_ln = local_net(ns).max(dim=1)[1].unsqueeze(dim=-1)

                    #use local network argmax and target network value
                    ns_target_max_val = torch.gather(ns_target_vals_tn, 1, ns_target_max_arg_ln)
                else:
                    # use local network to get value
                    ns_target_vals_ln = local_net(ns)
                    # use target network to get argmax
                    ns_target_max_arg_tn = target_net(ns).max(dim=1)[1].unsqueeze(dim=-1)

                    #use target network argmax and local network value
                    ns_target_max_val = torch.gather(ns_target_vals_ln, 1, ns_target_max_arg_tn)
            else:
                print(a.shape)
                # use target network only for value and argmax
                ns_target_max_val = target_net(ns).max(dim=1)[0].unsqueeze(dim=-1)

            assert(ns_target_max_val.requires_grad == False)

            td_targets = r + ((1-d) * GAMMA * ns_target_max_val)

        ###### TD CURRENT #######
        if isLearning: # if it is under learning mode need backprop
            local_net.train()
            td_currents_vals = local_net(s)

            td_currents = torch.gather(td_currents_vals, 1, a)
        else:
            local_net.eval()
            with torch.no_grad():
                td_currents_vals = local_net(s)

                td_currents = torch.gather(td_currents_vals, 1, a)

        local_net.train() #resume training for local network

        return td_targets, td_currents


    def step(self, state, action, reward, next_state, done, ep_prgs=(0,100)):
        """ handle memory update, learning and target network params update"""
        """
        epoche_status: destinated final epoche - current epoche
        """
        # internal rourtine
        def toBatchDim(v):
            return torch.from_numpy(v).unsqueeze(0)

        # get the td values to compute td errors
        td_target, td_current = self.get_TD_values(self.qnetwork_local,
                                                   self.qnetwork_target,
                                                   toBatchDim(state),
                                                   toBatchDim(np.array(action)).reshape(1,1),
                                                   reward,
                                                   toBatchDim(next_state),
                                                   done,
                                                   isLearning=False)

        # store the abs magnitude of td error, add eps to make sure it is non-zero
        td_error = torch.abs(td_target - td_current).cpu().numpy()

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, td_error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1

        # gradually increase beta to 1 until end of epoche
        if self.isTraining:
            self.p_replay_beta = min(1.0, self.p_replay_beta + 0.001)
            #self.p_replay_beta = P_REPLAY_BETA+((1-P_REPLAY_BETA)/ep_prgs[1])*ep_prgs[0]

        # If enough samples are available in memory, get random subset and learn
        if self.t_step >= REPLAY_MIN_SIZE:
            # training starts!
            if self.isTraining == False:
                self.print_params()
                print("Prefetch completed. Training starts!                         \r")
                self.isTraining = True

            #for i in range(LEARNING_LOOP): #greedy learning loop
            if USE_TREE:
                experiences, weight, ind = self.memory.sample_tree(self.p_replay_beta)
            else:
                experiences, weight, ind = self.memory.sample(self.p_replay_beta)

            self.learn(experiences, weight, ind, GAMMA)

            if self.t_step % UPDATE_EVERY == 0:
                # ------------------- update target network ------------------- #
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action =  np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action


    def learn(self, experiences, weight, ind, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            ind: index of memory being chosen, for TD errors update
            weight: the weight for loss adjustment because of priority replay
        """
        states, actions, rewards, next_states, dones = experiences

        td_targets, td_currents = self.get_TD_values(self.qnetwork_local,
                                                     self.qnetwork_target,
                                                     states, actions,
                                                     rewards, next_states,
                                                     dones,
                                                     isLearning=True)

        squared_err = torch.abs(td_currents - td_targets)**2
        loss = weight * squared_err
        loss = loss.mean()

        if self.lr_decay and self.t_step >= LR_DECAY_START:
            self.scheduler.step() #decay lr

        self.optimizer.zero_grad()
        loss.backward()
        # update the parameters
        self.optimizer.step()

        # update the td error in memory
        with torch.no_grad():
            td_errors_update = np.array([torch.abs(td_targets - td_currents).cpu().numpy()])
        self.memory.update(td_errors_update, ind)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, td_eps, seed,
                 p_replay_alpha, reward_scale=False, error_clip=False,
                 error_max=1.0, error_init=False, use_tree=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            td_eps: (float): to avoid zero td_error
            p_replay_alpha (float): discount factor for priority sampling
            reward_scale (flag): to scale reward down by 10
            error_clip (flag): max error to 1
            seed (int): random seed
        """
        self.useTree = use_tree
        self.memory = deque(maxlen=buffer_size)
        self.tree = SumTree(buffer_size) #create tree instance
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.td_eps = td_eps
        self.experience = namedtuple("Experience", field_names=["state", "action",
                                     "reward", "next_state", "done", "td_error"])
        self.seed = random.seed(seed)
        self.p_replay_alpha = p_replay_alpha
        self.reward_scale = reward_scale
        self.error_clip = error_clip
        self.error_init = error_init
        self.error_max  = error_max

        self.memory_index = np.zeros([self.buffer_size,1]) #for quicker calculation
        self.memory_pointer = 0

    def add(self, state, action, reward, next_state, done, td_error):
        """Add a new experience to memory.
        td_error: abs value
        """

        #reward clipping
        if self.reward_scale:
            reward = reward/10.0 #scale reward by factor of 10

        #error clipping
        if self.error_clip: #error clipping
            td_error = np.clip(td_error, -self.error_max, self.error_max)

        # apply alpha power
        td_error = (td_error ** self.p_replay_alpha) + self.td_eps

        # make sure experience is at least visit once
        if self.error_init:
            td_mad = np.max(self.memory_index)
            if td_mad == 0:
                td_error = self.error_max
            else:
                td_error = td_mad

        e = self.experience(np.expand_dims(state,0), action, reward,
                            np.expand_dims(next_state,0), done, td_error)
        if self.useTree:
            self.tree.add(td_error, e) # update the td score and experience data
        else:
            self.memory.append(e)

        ### memory index ###
        if self.memory_pointer >= self.buffer_size:
            #self.memory_pointer = 0
            self.memory_index = np.roll(self.memory_index, -1)
            self.memory_index[-1] = td_error #fifo
        else:
            self.memory_index[self.memory_pointer] = td_error
            self.memory_pointer += 1

    def update(self, td_updated, index):
        """
        update the td error values while restoring orders
        td_updated: abs value; np.array of shape 1,batch_size,1
        index: in case of tree, it is the leaf index
        """
        td_updated = td_updated.squeeze() # (batch_size,)

        #error clipping
        if self.error_clip: #error clipping
            td_updated = np.clip(td_updated, -1.0, 1.0)

        # apply alpha power
        td_updated = (td_updated ** self.p_replay_alpha) + self.td_eps

        ### checking memory and memory index are sync ###
        #tmp_memory = copy.deepcopy(self.memory)

        for i in range(len(index)):
            if self.useTree:
                #data_index = index[i]
                #tree_index = data_index + self.buffer_size - 1
                self.tree.update(index[i], td_updated[i])
            else:
                self.memory.rotate(-index[i]) # move the target index to the front
                e = self.memory.popleft()

                td_i = td_updated[i].reshape(1,1)

                e1 = self.experience(e.state, e.action, e.reward,
                                     e.next_state, e.done, td_i)

                self.memory.appendleft(e1) #append the new update
                self.memory.rotate(index[i]) #restore the original order

                ### memory index ###
                self.memory_index[index[i]] = td_i

                # make sure its updated
                # assert(self.memory[index[i]].td_error == self.memory_index[index[i]])
            ### checking memory and memory index are sync ###
            #for i in range(len(self.memory)):
            #    assert(self.memory_index[i] == self.memory[i].td_error)
            #    if i in index:
            #        assert(td_updated[list(index).index(i)] == self.memory[i].td_error)
            #    else:
            #        print(self.memory[i].td_error)
            #        assert(tmp_memory[i].td_error == self.memory[i].td_error)



    def sample(self, p_replay_beta):
        """Sample a batch of experiences from memory."""
        l = len(self.memory)
        p_dist = (self.memory_index[:l]/np.sum(self.memory_index[:l])).squeeze()

        assert(np.abs(np.sum(p_dist) - 1) <  1e-5)
        assert(len(p_dist) == l)

        # get sample of index from the p distribution
        sample_ind = np.random.choice(l, self.batch_size, p=p_dist)

        experiences = [] #faster to avoid indexing

        ### checking: make sure the rotation didnt screw up the memory ###
        #tmp_memory = copy.deepcopy(self.memory) #checking

        # get the selected experiences: avoid using mid list indexing
        es, ea, er, en, ed = [], [], [], [], []
        for i in sample_ind:
            self.memory.rotate(-i)
            e = copy.deepcopy(self.memory[0])
            es.append(e.state)
            ea.append(e.action)
            er.append(e.reward)
            en.append(e.next_state)
            ed.append(e.done)
            experiences.append(copy.deepcopy(self.memory[0]))
            self.memory.rotate(i)

        ### checking: make sure the rotation didnt screw up the memory ###
        #for i in range(len(tmp_memory)):
        #    assert(tmp_memory[i].td_error == self.memory[i].td_error) #checking

        states = torch.from_numpy(np.vstack(es)).float().to(device)
        actions = torch.from_numpy(np.vstack(ea)).long().to(device)
        rewards = torch.from_numpy(np.vstack(er)).float().to(device)
        next_states = torch.from_numpy(np.vstack(en)).float().to(device)
        dones = torch.from_numpy(np.vstack(ed).astype(np.uint8)).float().to(device)

        # for weight update adjustment
        selected_td_p = p_dist[sample_ind] #the prob of selected e

        ### checker: the mean of selected TD errors should be greater than
        ### checking: the mean of selected TD err are higher than memory average
        if p_replay_beta > 0:
            assert(np.mean(self.memory_index[sample_ind]) >= np.mean(self.memory_index[:l]))

        #weight = (np.array(selected_td_p) * l) ** -p_replay_beta
        #max_weight = (np.min(selected_td_p) * self.batch_size) ** -p_replay_beta

        weight = (1/selected_td_p * 1/l) ** p_replay_beta
        weight =  weight/np.max(weight) #normalizer by max
        weight = torch.from_numpy(np.array(weight)).float().to(device) #change form
        assert(weight.requires_grad == False)

        return (states, actions, rewards, next_states, dones), weight, sample_ind

    def sample_tree(self, p_replay_beta):
        n = self.batch_size
        # Create a sample array that will contains the minibatch
        e_s, e_a, e_r, e_n, e_d = [], [], [], [], []

        sample_ind = np.empty((self.batch_size,), dtype=np.int32)
        weight = np.empty((self.batch_size, 1))

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        td_score_segment = self.tree.total_td_score / self.batch_size  # priority segment

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.buffer_size:]) / self.tree.total_td_score
        if p_min == 0:
            p_min = self.td_eps # avoid div by zero
        max_weight = (p_min * self.buffer_size) ** (-p_replay_beta)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = td_score_segment * i, td_score_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            leaf_index, td_score, data = self.tree.get_leaf(value)

            #P(j)
            sampling_p = td_score / self.tree.total_td_score

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            weight[i, 0] = np.power(self.batch_size * sampling_p, -p_replay_beta)/ max_weight

            sample_ind[i]= leaf_index

            e_s.append(data.state)
            e_a.append(data.action)
            e_r.append(data.reward)
            e_n.append(data.next_state)
            e_d.append(data.done)

        states = torch.from_numpy(np.vstack(e_s)).float().to(device)
        actions = torch.from_numpy(np.vstack(e_a)).long().to(device)
        rewards = torch.from_numpy(np.vstack(e_r)).float().to(device)
        next_states = torch.from_numpy(np.vstack(e_n)).float().to(device)
        dones = torch.from_numpy(np.vstack(e_d).astype(np.uint8)).float().to(device)

        weight = torch.from_numpy(weight).float().to(device) #change form

        return (states, actions, rewards, next_states, dones), weight, sample_ind

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
