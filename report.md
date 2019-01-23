
# Learning Summary on Implementing DQN for Unity ML-Agents - Bananas Collector

### Implementation Details

#### A 2-layer DQN
Instead of using other techniques like discretization of continuous states, this project used a neural network as a function approximator to estimate the Q value of state. The network being finally chosen was a 2-layer network: first one is of 128 nodes, followed by rulu activation and the second one is 64 nodes (no activation). A lot of different combinations have been tried and it seems that a simpler network tends to work better.

#### Double Network
After a few attempts, it is found that double network have the following benefits:
- accelerate the learning. In my experience, i can achieve 13 reward faster (653 episodes) than a network without double network (1009 episodes).
- make the td values magnitude smaller
Here 50% of chance the target network will choose the action and local network will calculate the Q value. Another 50% of chance their role will swap. I personally find it working well for this task.

#### Prioritized Experience Replay (PER)
PER was implemented in this project and it took me so much time to do it. There are a couple of components of PER being implemented here:
- A Binary Tree architecture that used a one-dimensional array to store the td errors values. This data structure allows very easy and quick access to values and to do sorting. The implementation does NOT have any recursion as, after some testing, it is found that even a memory size of 1e4 could cause recursion number to exceed python's max limit. This structure is very efficient compared to deque. I also implemented deque here so they can be compared.
- There are 3 parameters used in PER.
![PER formula](https://github.com/chihoxtra/dqn_unity_navigation/blob/master/per_formula.png)
  - Alpha: the power value applied when calculating the probability. Value used here is 0.6 as I personally find that this works the best.
  - Beta: 0.5 and it grows linearly to 1 as mentioned in the paper.
  - eps: This is added to all powered td values to make sure that they are not zero. I personally find that it cannot be too small cause otherwise some experience would never be chosen. Here I used 1e-3.
- Weight adjustment was very tricky. The calculation could be different if we take the mean first before multiplying by the squared difference between td target and td current.
- To facilitate training, the memory is 'pre-fetched' with experience before the training started. This could make the training smoother as different experience scenarios are available at earlier stage and it seems that it is always better for agent to encounter these special cases earlier instead of later when the network is already taking shape.
- PER was found to be able to accelerate the learning.
<p>
Here are a summary of the hyper parameters used:
Parameters | Value
------------ | -------------
Memory buffer size  | 1e5        
batch size  |  64
REPLAY_MIN_SIZE  |  int(1e5)   
Gamme  | 0.99                  
Tau  | 1e-3                    
Learning Rate = 1e-4  
update target network frequency  | 16    
minimal TD error  |  1e-3         
PER alpha = 0.6          
PER beta  |  0.4     

#### The Result:
After soooooo many different trial and errors, I am glad that I am finally able to reach an average score of 13 within 653 training episodes. <P>
![Reward Plot with double network](https://github.com/chihoxtra/dqn_unity_navigation/blob/master/reward_withdoublenetwork1.png)
<P>
![Reward Plot without double network](https://github.com/chihoxtra/dqn_unity_navigation/blob/master/reward_plot.png)
<P>
Here is a video of the trained agent:<br>
![Trained Model](https://github.com/chihoxtra/dqn_unity_navigation/blob/master/youtube.png)<br>
[Video Link](https://www.youtube.com/watch?v=63gxOq67coM&feature=youtu.be)

#### Other attempts:
- *Dual Network* an attempt to use duel network is also made. However, unlike the case in Atari game play agent (by google deepmind) where the convolutional part of the network is shared, the base network used here is very shallow and I find that if only a small portion of the network is shared, the performance might be worse compared to the Atari case. Therefore even after the implementation, it is not used at the end.
- *Initial error value* some implementations set the initial error value of an experience to max of all td values available when applying PER. The idea here is that a low TD error value might not necessarily means that the experience is not important. By setting the initial TD value of an experience to max it guaranteed that at least once would the experience be sampled. The draw back here is that it could potentially wrongly boost the priorities of some less important experience that would otherwise not be chosen. I personally find that this is not working well in this case. Maybe TD error itself is already a very good approximation of the importance of experience and states and so an artificial push is not necessary here.

#### Future Ideas:
- I personally found the implementation of DQN very powerful and interesting. I have already started working on a [Ms Pacman from OpenAI environment](https://github.com/chihoxtra/dqn_ms_pacman  by taking pixel/screenshots inputs. Hopefully I could finish it soon.
- Other interesting ideas that I cant wait to explore are:
  - prioritize experience based on a hybrid of factors like reward and td errors?
  - how about multi-agent env? I found an example of using RL to train an agent to play StarCraft. That would be a very interesting project.
Cant wait to explore more!
