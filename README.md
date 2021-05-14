[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


#  Continuous Control

### Introduction

This project attempts to solve the Tennis environment from Unity technologies
 [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis) .

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

## Installation

* tennis.py is the landing script
* This repo contains contains :
	1. The agent (ddpg_agent.py) that implements the DDPG RL agent functionalities   
	2. Multi Agent Actor and Critic Models : model.py
	
## Requirements:
	* Python 3.6 or greater
	* pythorch 1.7.1
	* Unity ML Agents, Tennis Environment

## Running the code
* Set the parameters in the params object in continuous_control.py
* Run the tennis.py
* By default, the script ends when a score of 30 is met.
* Three files are generated:
	* The pickle file containing the parameters, scores. 
	* Checkpoint of the actor and critic NN model.
## Baseline results
The default parameter/results are located in the folder '/Results'. They can be analyzed with the plotres.py utility.
Checkpoint files are located in the checkpoint folder/


