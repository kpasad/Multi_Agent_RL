from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
from paramutils import *
from collections import deque
import time
import pickle as pk
import torch

params=parameters()
params.op_filename_prefix="Tennis_"+time.strftime("%H_%M_%S",time.gmtime(time.time()))

params.n_episodes=200
params.OU_noise_sigma=0.1
params.OU_noise_mu=0
params.OU_noise_theta=0.15

env = UnityEnvironment(file_name="C:/Users/kpasad/Downloads/Tennis_Windows_x86_64/Tennis_Windows_x86_64/Tennis.exe",no_graphics=True)



# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]




# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from ddpg_agent import MADDPG
agent = MADDPG(24, 2, 2, 0,params)

scores_window = deque(maxlen=100)  # save last 100 total scores in one episode
all_scores = []
avg_scores_window = []
max_score = 0  # save best score in that run
n_episodes=5000



for i_episode in range(1, n_episodes + 1):

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state

    agent.reset()  # reset the agent
    scores = np.zeros(num_agents)  # initialize the score

    while True:
        actions = agent.act(states)  # select an action from one agent
        env_info = env.step(actions)[brain_name]  # perform the action

        next_states = env_info.vector_observations  # get next state
        rewards = env_info.rewards  # get reward
        dones = env_info.local_done  # check done

        agent.step(states, actions, rewards, next_states, dones, num_updates=3)  # agent step

        states = next_states
        scores += rewards

        if np.any(dones):
            break

            # score for one episode of mean of all agents
    avg_score = np.mean(scores)

    # save last 100 avg_score scores
    #scores_window.append(avg_score)
    #all_scores.append(avg_score)
    scores_window.append(np.max(scores))
    all_scores.append(np.max(scores))


    avg_scores_window.append(np.mean(scores_window))
    noise_damp = np.mean(scores_window)

    if i_episode % 10 == 0:
        print('Episode {}\tMax Reward: {:.3f}\tAverage Reward: {:.3f}'.format(
            i_episode, np.max(scores), np.mean(scores_window)))

    if np.mean(scores_window) >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(
            i_episode - 100, np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_0.pth')
        torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_0.pth')

        break

pk.dump([all_scores,avg_scores_window, params],open(params.op_filename_prefix+'.pk','wb'))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(all_scores)+1), all_scores)
plt.plot(np.arange(1, len(avg_scores_window)+1), avg_scores_window)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
