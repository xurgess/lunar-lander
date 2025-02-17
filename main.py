import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
from concurrent.futures import ProcessPoolExecutor
import pickle
import multiprocessing
import logging
import sys

class Actor(nn.Module):
    def __init__(self, state_dimensions, action_dimensions, max_action):
        super(Actor, self).__init__()

        # first layer of the neural network, state_dimension amt of neurons to 256 neurons
        self.first_layer = nn.Linear(state_dimensions, 256)


        self.second_layer = nn.Linear(256, 256)

        # 256 neurons to action_dimension amt of neurons
        self.third_layer = nn.Linear(256, action_dimensions)

        # the scale of the action (-1, 1) for our purposes
        self.max_action = max_action


    def forward(self, state):
        # returns a "tensor" (a matrix with a single column) representing the output of the first layer
        x = torch.relu(self.first_layer(state))
        x = torch.relu(self.second_layer(x))

        # returns a tensor with one column and eight rows, each item represents the
        # continous action to take on a given control (left axis, right axis, etc)
        action = torch.tanh(self.third_layer(x)) * self.max_action
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dimensions, action_dimensions):
        super(Critic, self).__init__()

        # a tad different from the neural network setup in the actor, instead of taking in states and outputting actions
        # we are taking in a state and an action while outputting a single number (the Q-value)
        self.first_layer = nn.Linear(state_dimensions + action_dimensions, 256)
        self.second_layer = nn.Linear(256, 256)
        self.third_layer = nn.Linear(256, 1)

    def forward(self, state, action):

        # takes the 8 pieces of the state and the 2 pieces of the action and shapes it into a tensor with 10 neurons
        # these neurons altogether tell the network the state and the action taken in that state
        x = torch.cat([state, action], 1)

        # then we run through the neural network
        x = torch.relu(self.first_layer(x))
        x = torch.relu(self.second_layer(x))

        # eventually ending up with a single value (the Q-value)
        value = self.third_layer(x)
        return value
    
# the purpose of this replay buffer is to further randomize data
# primarily for the purpose of eliminating learning by correlation,
# but also it serves as quite the computational speedup, as the "batches"
# that we run through the critic and the actor get parallelized
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # we gather a batch_size amount of random snapshots from the buffer
        batch = random.sample(self.buffer, batch_size)

        # and we return them as numpy arrays to be tested on
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
batch_size = 64 # the amount of snapshots to be run through the critic at once
gamma = 0.99  # the discount factor
update_rate = 0.001  # the learning rate
target_update_rate = 0.0005  # the update rate rate of the target network
exploration_noise = 0.1 # the rate at which random actions occur

# initialize environment
env = gym.make("LunarLanderContinuous-v2")
state_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.shape[0]
max_action = env.action_space.high[0]

# initialize networks
actor = Actor(state_dimensions, action_dimensions, max_action)
critic = Critic(state_dimensions, action_dimensions)
target_actor = Actor(state_dimensions, action_dimensions, max_action)
target_critic = Critic(state_dimensions, action_dimensions)

# copy weights to target networks
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# initialize optimizers, to tune the weights of the networks
actor_optimizer = optim.Adam(actor.parameters(), lr=update_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=update_rate)

# initialize replay buffer
replay_buffer = ReplayBuffer(max_size=1000000)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def run_simulation(batch_size, update_rate, exploration_noise, test_index):
    # initialize environment
    env = gym.make("LunarLanderContinuous-v2")
    state_dimensions = env.observation_space.shape[0]
    action_dimensions = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # initialize networks
    actor = Actor(state_dimensions, action_dimensions, max_action)
    critic = Critic(state_dimensions, action_dimensions)
    target_actor = Actor(state_dimensions, action_dimensions, max_action)
    target_critic = Critic(state_dimensions, action_dimensions)

    # target update rate is half of the update rate
    target_update_rate = update_rate / 2

    logger.info(f"Running test_{test_index}: batch_size={batch_size}, update_rate={update_rate}, exploration_noise={exploration_noise}")

    # copy weights to target networks
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # initialize optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=update_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=update_rate)

    # initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=1000000)

    ### for graphs and stuff
    episode_rewards = []
    current_Q_means = []
    target_Q_means = []
    actor_losses = []
    critic_losses = []
    ###

    # number of episodes to run
    num_episodes = 1000
    episode_reward_dict = {}

    for episode in range(num_episodes):

        # reset environment
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:

            # actor selects an action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_tensor).detach().numpy()[0]

            # add exploration noise
            action = action + np.random.normal(0, exploration_noise, size=action_dimensions)

            # ensure action is within bounds
            action = np.clip(action, -max_action, max_action)

            # take action in environment
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward

            # store transition in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # update state
            state = next_state

            # if buffer is large enough, start training
            if len(replay_buffer.buffer) > batch_size:
                # sample from buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # convert to tensors
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # calculate target Q es
                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_Q = target_critic(next_states, next_actions)
                    target_Q = rewards + (1 - dones) * gamma * target_Q
                    target_Q_means.append(target_Q.mean().item())

                # calculate current Q values and critic loss
                current_Q = critic(states, actions)
                current_Q_means.append(current_Q.mean().item())
                critic_loss = torch.nn.MSELoss()(current_Q, target_Q)
                critic_losses.append(critic_loss.item())

                # update critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # calculate actor loss and update actor
                actor_loss = -critic(states, actor(states)).mean()
                actor_losses.append(actor_loss.item())
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update target networks
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(target_update_rate * param.data + (1 - target_update_rate) * target_param.data)

        episode_rewards.append(episode_reward)
        episode_reward_dict[episode] = episode_reward
        if episode % 100 == 0:
            logger.info(f"test {test_index}: episode {episode}, reward: {episode_reward}")

    # save the model
    torch.save(actor.state_dict(), f'test2_{test_index}_actor.pth')

    # save the data
    data_to_save = {
        "episode_rewards": episode_rewards,
        "current_Q_means": current_Q_means,
        "target_Q_means": target_Q_means,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses
    }

    with open(f'test2_{test_index}.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
        

    env.close()
    return f"completed test {test_index}"

# this will run parallel execution using joblib
if __name__ == "__main__":

    ''' these were the first set of hyper params test
    batch_sizes = [16, 32, 64, 128]
    update_rates = [0.0001, 0.0005, 0.001, 0.005]
    exploration_noises = [0.05, 0.1, 0.2, 0.3]
    '''

    # but, given the results, i wanna test a few more :O
    batch_sizes = [16, 32, 64]
    update_rates = [0.01, 0.05, 0.1]
    exploration_noises = [0.001, 0.005, 0.01]

    # first we prepare the list of tasks
    tasks = []
    test_index = 0
    for x in range(len(batch_sizes)):
        for y in range(len(update_rates)):
            for z in range(len(exploration_noises)):
                test_index += 1
                tasks.append((batch_sizes[x], update_rates[y], exploration_noises[z], f"{x}{y}{z}"))

    print(f"spun up {len(tasks)} processes")

    # then, we run in parallel each task
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.starmap(run_simulation, tasks)

    for result in results:
        print(result)


