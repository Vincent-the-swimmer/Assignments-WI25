import numpy as np
from tqdm import tqdm
from .discretize import quantize_state, quantize_action
from dm_control.rl.control import Environment

# Initialize the Q-table with dimensions corresponding to each discretized state variable.
def initialize_q_table(state_bins: dict, action_bins: list) -> np.ndarray:
    """
    Initialize the Q-table with dimensions corresponding to each discretized state variable.

    Args:
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.

    Returns:
        np.ndarray: A Q-table initialized to zeros with dimensions matching the state and action space.
    """
    # TODO: Implement this function
    dim_state = []
    for i in range(len(state_bins["position"])):
        dim_state.append(len(state_bins["position"][i]))
    for i in range(len(state_bins["velocity"])):
        dim_state.append(len(state_bins["velocity"][i]))
    dim_action = len(action_bins)
    dim_all = tuple(dim_state) + (dim_action, )
    
    return np.zeros(dim_all)
    ...
    


# TD Learning algorithm
def td_learning(env: Environment, num_episodes: int, alpha: float, gamma: float, epsilon: float, state_bins: dict, action_bins: list, q_table:np.ndarray=None) -> tuple:
    """
    TD Learning algorithm for the given environment.

    Args:
        env (Environment): The environment to train on.
        num_episodes (int): The number of episodes to train.
        alpha (float): The learning rate. 
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.
        q_table (np.ndarray): The Q-table to start with. If None, initialize a new Q-table.

    Returns:
        tuple: The trained Q-table and the list of total rewards per episode.
    """
    if q_table is None:
        q_table = initialize_q_table(state_bins, action_bins)
        
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # reset env
        # run the episode
            # select action
            # take action
            # quantize the next state
            # update Q-table
            # if it is the last timestep, break
        # keep track of the reward

        action_taken, reward, discount, state = env.reset()
        state = tuple(quantize_state(state, state_bins))
        done = False
        total_reward = 0
        while not done:
            action = egreedy_action_selection(q_table, state, action_bins, epsilon)

            action_taken, reward, discount, next_state = env.step(action)

            next_state = tuple(quantize_state(next_state, state_bins))
            if reward == None:
                break
            next_action = egreedy_action_selection(q_table, next_state, action_bins, epsilon)
            q_table[state][action] = q_table[state][action] + alpha*(reward + gamma*q_table[next_state][next_action]- q_table[state][action])
            
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
            
    #print(q_table[1])
    # action_taken, reward, discount, next_state = env.step(action)


    return q_table, rewards

def egreedy_action_selection(q_table, state_index, action_bins, epsilon):
    if np.random.rand() < epsilon:
        best_action = np.random.randint(0, len(action_bins))
    else:
        check_array = q_table[state_index]
        best_action = np.argmax(check_array)
    return best_action

def greedy_policy(q_table: np.ndarray) -> callable:
    """
    Define a greedy policy based on the Q-table.    

    Args:
        q_table (np.ndarray): The Q-table from which to derive the policy.

    Returns:
        callable: A function that takes a state and returns the best action. 
    """
    def policy(state):
        return np.argmax(q_table[state])
    return policy