
# DQN for Unity ML-Agents - Bananas Collector

<p align="center"><a href="https://gym.openai.com/envs/MsPacman-v0/">
 <img width="500" height="269" src="https://github.com/chihoxtra/dqn_unity_navigation/blob/master/banana.gif"></a>
</p>

### Project Summary

This project attempts to use a DQN as a function appropximator to enable an agent to get as much reward as possible under the unity bananas collector environment.

### About the Environment and the Task

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. <br><br>
According to [Unity Github](https://github.com/Unity-Technologies/ml-agents/issues/1134) , states can be interpreted as follows:<br>
- Vector - 37 values<br>
  - Values 1-36 - ray values<br>
  - Value 37 - agent linear velocity<br>
Ray Values<br>
- 6 vectors of length 6<br>
  - Values 1-5 - (1 or 0) - ray segments in increasing distance from the agent presence of a banana<br>
  - Value 6 - Angular rotation of the ray from it starting point

Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:<br>
* 0 - move forward.<br>
* 1 - move backward.<br>
* 2 - turn left.<br>
* 3 - turn right.<br>
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Packages Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
And here is the list of packages requirements:
- tensorflow==1.7.1
- Pillow>=4.2.1
- matplotlib
- numpy>=1.11.0
- jupyter
- pytest>=3.2.2
- docopt
- pyyaml
- protobuf==3.5.2
- grpcio==1.11.0
- torch==0.4.0
- pandas
- scipy
- ipykernel

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

6. Make sure you have the right Unity env to run this notebook. If you do not wish to install the whole unity environment, you can choose to download the 'self-contained' environments thru the following links:
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- [AWS Linus](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip): this is a headless version and please remember to [turn on Xorg](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md) as the env needs to output to a display channel to run. Note that the codes should be fully compatable with GPU enabled environment. You might however need to enable GPU by installing GPU drivers in AWS.

7. To run the notebook, make sure you are in the environment 'drlnd'. You can activate your environment like this:
```bash
source activate drlnd
```
then you can start the notebook on your local machine.
```bash
jupyter notebook --ip=0.0.0.0 --no-browser
```
To run the notebook, press shift-enter and it shall run the codes cell by cell.

Any question? please feel free to contact me at: [samuelpun@gmail.com](mailto:samuelpun@gmail.com)
