# Unity-ML Tennis

This repo contains a working code for solving the Unity ML Agent called Tennis using a Multi-Agent DDPG  Network.

In this environment, two agents play Tennis against each other making it suitable to experiment multi-agent reinforcement solutions.

## The Task
 The agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
-   This yields a single  score  for each episode.

## The Environment and the Agent
### The State Space
A single time step contains 8 float values, representing the position and velocity of the ball and the rackets, so 3 frames concatenated resulting in a vector with 24 numbers.
In this vector there are information about both agents. Each agent has separate state so the whole state given by environment is [2, 24].

In this environment, two agents (players) control rackets to bounce a ball over a net and they need to know in which position and velocity the rackets and the ball are moving

They observation space comprises the image of the current state of the game. A possible solution could be to add the past image to the observation space. So this temporal correlation states the change in position and velocity of the ball and the rackets

Remember the case of DQN for atari, in which you had to stack the last 4 frames of a history to produce the input to the Q-function and have a good state representation that could solve the problem.

This strategy is useful in situations where there is important information about what has happened in the past, or happened over time which you want to keep track of, but don't necessarily have a good representation for. Like the position and velocity of the ball and the rackets.

The "Number of stacked" parameter changes how many sets of observations into the past you'd like to stack. Increasing this allows the agent to "see" further into the past.

### The Action Space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.


## The Solution: Multiagent DDPG
There are several multi-agent architectures possible. The architecture for this project contains:
1. Actors that share weights
2. Critic that concatenates the state and action space of both  the  agents.
It might seem that the combination of 1 and 2 reduces the architecture to a single agent because both the actor and the critic operate on state and state+action space, respectively, of both players. However, there is a subtle diffrence. The actors share weight, but their inputs are state space of a single player. This is different from a single actor whose input is combined state of both the player.

## DDPG algorithm
DDPG algorithm was proposed in [this](https://arxiv.org/pdf/1509.02971.pdf) paper.
Q-Learning cannot be applied to continuous action space as the optimization of the action (argmax) is inefficient if not immposible. DDPG solves this problem by directly mapping the state to the optimal action. The mapping is a non linear function, that is typically implemented as a neural network.  Exploration is invoked by dithering the action with Ornstein-Uhlenbeck noise.

DDPG is often bucketed in family of actor-critic algorithm.The actor implements a current policy to map the state to optimal action. The critic implemets the Q function estimation, and estimates the Action value given a state and an action. The actor is trained by the maximising the gradient of action value. Critic is trained by minimizing the MSE between predicted Q-value and meausured Q value (as a surrogate for true long term).

Experinces are stored in a replay buffer. 
In order to encourage exploration during training, Ornstein-Uhlenbeck noise is added to the actors selected actions.

### Network size 

The actor network are:
|Layer|Input size  |Output(Number of Neurons)|Nonlinearity|
|--|--|--|--|
|1|24|128| ReLu|
|2 |128|64|ReLu|
|3|64|2|Linear|

The critic network are:
|Layer|Input size  |Output(Number of Neurons)|Nonlinearity|
|--|--|--|--|
|1|48|128| ReLu|
|2 |128+4|64|ReLu|
|3|64|1|Linear|

The critic network concatenates the states from the two players. After the first layer
the critic network concatenates the action estimated by the action network with a feature representation of the state instead of concatenating directly with the state. Critic learns faster on the feature representation instead of the raw state.

Key parameters are::
|Parameter	|Value|
|--|--|
|replay buffer size|	int(1e6)|
|minibatch size	|256|
|discount factor	|0.99|
|tau (soft update)|	1e-3|
|learning rate actor|	1e-5|
|learning rate critic|	1e-5|
|L2 weight decay|	0|
|UPDATE_EVERY	|20|
|NUM_UPDATES	|10|
|EPSILON	|1.0|
|EPSILON_DECAY|	1e-6|
|NOISE_SIGMA	|0.1|
|Gamma| 1|
Not in particular, that Gamma is set to 1

## Running the simulation
Run the file Tennis.py. The simulation will terminate when one of the two condition is met: The agent gets a score of 13 or the predetermined number of episodes elapse. At the end of simulation, a pickle file is generated. The file contains a dump of parameters as the raw scores. The scores can be analyzed to create the plot below using the utility script plotres.py. The network weights are checkpointed as well.

## Performance
DDPG is notoriously difficult to train. The networks learn in successfully in a very narrow range of hyperparameter.
1. Exploration model: The action determined by Action network is perturbed by the Ornstein–Uhlenbeck noise. Training the networks required change to the noise model, to change from uniform random to normal distribution

#dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])  
dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.standard_normal() for i in range(len(x))]) 
With the parameters mentioned above, training progressed at a very slow rate for approximately initial 35% of the total time. In contrast to a steady accumulation of rewards, singular large reward events propelled the agents towards successfully meeting the rewards target.
![Scores for various learning rate](https://github.com/kpasad/Compettion_and_colaboration_MADDPG/blob/main/Results/final_scores.jpeg)

## Conclusion
There are several minor and modification to try:
1. OU- Noise annealing: RL training is very sensitive to the exploration-exploitation tradeoff. Noise annealing can automate this to some extent.
2. Here both agents share weights. In order to achieve diversity, the two agents may  begin with different random initialization and learn their own trajectories. At certain events (e.g. high reward state-actions) or after certain intervals, they can share the weights.