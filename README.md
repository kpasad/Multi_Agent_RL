[//]: # (Image References)

[image1]: https://github.com/kpasad/Multi_Agent_RL/blob/main/results/agents_play_tennis.gif "Trained Agents playing Tennis"



#  Continuous Control

### Introduction

This project attempts to solve the Tennis environment from Unity technologies
 [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis) .

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
The problem is solved when the average of the maximum rewards across agents is 0,5 over 100 episodes.

See Report.md for detailed description

## Installation

* tennis.py is the landing script
* This repo contains contains :
	1. The agent (ddpg_agent.py) that implements the DDPG RL agent functionalities   
	2. Multi Agent Actor and Critic Models : model.py
	
## Requirements:
	* Python 3.6 or greater
	* pythorch 1.7.1
	* Unity ML Agents, Tennis Environment
	
## Step 1: Instantiating the Python environment and dependencies
This installs all the required dependencies.
Please see the instruction [ Udacity DRLND GitHub repository.](https://github.com/udacity/deep-reinforcement-learning#dependencies)

## Step 2: Installing the Tennis Environment
Please see the instruction [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started)	
	

## Running the code
* Set the parameters in the params object in Tennis.py
* Run the tennis.py
* By default, the script ends when a score of 0.5 is met.
* Three files are generated:
	* The pickle file containing the parameters, scores. 
	* Checkpoint of the actor and critic NN model.
## Baseline results
The default parameter/results are located in the folder '/Results/Tennis.pk'. 
Model chekpoints are located in the files '/Results/checkpoint_actor_0.pth' and '/Results/checkpoint_actor_1.pth'. 
The progress of simulation is captured in '/Results/output_screenShot.txt'



