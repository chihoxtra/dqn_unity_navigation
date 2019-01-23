
# Learning Summary on Implementing DQN for Unity ML-Agents - Bananas Collector

### Implementation Details

#### A 2-layer DQN
Instead of using other techniques like discretization of continuous states, this project used a neural network as a function approximator to estimate the Q value of state. The network being finally chosen was a 2-layer network: first one is of 128 nodes, followed by rulu activation and the second one is 64 nodes (no activation). A lot of different combinations have been tried and it seems that a simpler network tends to work better.

#### Prioritized Experience Replay (PER)
PER was implemented in this project and it took me so much time to do it. There are a couple of components of PER being implemented here:
- A Binary Tree architecture that used a one-dimensional array to store the td errors values. This data structure allows very easy and quick access to values and to do sorting. The implementation does NOT have any recursion as, after some testing, it is found that even a memory size of 1e4 could cause recursion number to exceed python's max limit. This structure is very efficient compared to deque. I also implemented deque here so they can be compared.
- There are 3 parameters used in PER.
  - Alpha: the power value applied when calculating the probability. Value used here is 0.6 as I personally find that this works the best.
  - Beta: 0.5 and it grows linearly to 1 as mentioned in the paper.
  - eps: This is added to all powered td values to make sure that they are not zero. I personally find that it cannot be too small cause otherwise some experience would never be chosen. Here I used 1e-3.
- Weight adjustment was very tricky. The calculation could be different if we take the mean first before multiplying by the squared difference between td target and td current.
- PER was found to be able to accelerate the learning.

#### Other attempts:
- *Dual Network* an attempt to use duel network is also made. However, unlike the case in Atari game play agent (by google deepmind) where the convolutional part of the network is shared, the base network used here is very shallow and I find that if only a small portion of the network is shared, the performance might be worse compared to the Atari case. Therefore even after the implementation, it is not used at the end.
- *Double Network* it is also attempted to use a double network. Here 50% of chance choice of action would be made by target network whereas Q value would be produced by local network. And 50% of the chance it will be vice versa. However, I also find that this is not exactly contributing a lot in this case. Maybe my implementation was wrong. I shall continue to review it.
- *Initial error value* some implementations set the initial error value of an experience to max of all td values available when applying PER. The idea here is that a low TD error value might not necessarily means that the experience is not important. By setting the initial TD value of an experience to max it guaranteed that at least once would the experience be sampled. The draw back here is that it could potentially wrongly boost the priorities of some less important experience that would otherwise not be chosen. I personally find that this is not working well in this case. Maybe TD error itself is already a very good approximation of the importance of experience and states and so an artificial push is not necessary here.
