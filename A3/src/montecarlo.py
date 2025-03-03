import numpy as np
import random
from collections import defaultdict
from src.racetrack import RaceTrack

class MonteCarloControl:
    """
    Monte Carlo Control with Weighted Importance Sampling for off-policy learning.
    
    This class implements the off-policy every-visit Monte Carlo Control algorithm
    using weighted importance sampling to estimate the optimal policy for a given
    environment.
    """
    def __init__(self, env: RaceTrack, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 1000):
        """
        Initialize the Monte Carlo Control object. 

        Q, C, and policies are defaultdicts that have keys representing environment states.  
        Defaultdicts (search up the docs!) allow you to set a sensible default value 
        for the case of Q[new state never visited before] (and likewise with C/policies).  
        

        Hints: 
        - Q/C/*_policy should be defaultdicts where the key is the state
        - each value in the dict is a numpy vector where position is indexed by action
        - That is, these variables are setup like Q[state][action]
        - state key will be the numpy state vector cast to string (dicts require hashable keys)
        - Q should default to Q0, C should default to 0
        - *_policy should default to equiprobable (random uniform) actions
        - store everything as a class attribute:
            - self.env, self.gamma, self.Q, etc...

        Args:
            env (racetrack): The environment in which the agent operates.
            gamma (float): The discount factor.
            Q0 (float): the initial Q values for all states (e.g. optimistic initialization)
            max_episode_size (int): cutoff to prevent running forever during MC
        
        Returns: none, stores data as class attributes
        """
         # Your code here
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q0 = Q0
        self.max_episode_size = max_episode_size
        self.Q = defaultdict(lambda: np.full(env.n_actions, 0))
        self.C = defaultdict(lambda: np.zeros(env.n_actions))
        self.greedy_policy = defaultdict(lambda: np.full(env.n_actions, 1 / env.n_actions))
        self.egreedy_policy = defaultdict(lambda: np.full(env.n_actions, 1 / env.n_actions))


    def create_target_greedy_policy(self):
        """
        Loop through all states in the self.Q dictionary. 
        1. determine the greedy policy for that state
        2. create a probability vector that is all 0s except for the greedy action where it is 1
        3. store that probability vector in self.target_policy[state]

        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # Your code here
        for state in self.Q:
            best_action = np.argmax(self.Q[state])
            action_prob = np.zeros(len(self.Q[state]))
            action_prob[best_action] = 1
            self.greedy_policy[state] = action_prob


    def create_behavior_egreedy_policy(self):
        """
        Loop through all states in the self.target_policy dictionary. 
        Using that greedy probability vector, and self.epsilon, 
        calculate the epsilon greedy behavior probability vector and store it in self.behavior_policy[state]
        
        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # for state in self.greedy_policy:
        #     #Do epsilon greedy action selection
        #     if np.random.rand() < self.epsilon:
        #         best_action = np.random.randint(0, len(self.greedy_policy[state]))
        #     else:
        #         best_action = np.argmax(self.greedy_policy[state])
            
        #     #Assigns the value to the egreedy policy
        #     action_prob = np.zeros(len(self.greedy_policy[state]))
        #     action_prob[best_action] = 1
        #     self.egreedy_policy[state] = action_prob   
        for state in self.Q:
            best_action = np.argmax(self.greedy_policy[state])
            action_prob = np.full(len(self.greedy_policy[state]), self.epsilon / len(self.greedy_policy[state]))
            action_prob[best_action] += 1 - self.epsilon
            self.egreedy_policy[state] = action_prob
   


        
    def egreedy_selection(self, state):
        """
        Select an action proportional to the probabilities of epsilon-greedy encoded in self.behavior_policy
        HINT: 
        - check out https://www.w3schools.com/python/ref_random_choices.asp
        - note that random_choices returns a numpy array, you want a single int
        - make sure you are using the probabilities encoded in self.behavior_policy 

        Args: state (string): the current state in which to choose an action
        Returns: action (int): an action index between 0 and self.env.n_actions
        """
        if not isinstance(state, tuple):
            state = tuple(state.tolist())
        if np.random.rand() < self.epsilon:
            best_action = np.random.randint(0, len(self.egreedy_policy[state]))
        else:
            best_action = np.argmax(self.egreedy_policy[state])

        return state, best_action
        # return np.random.choice(len(self.egreedy_policy[state]), p=self.egreedy_policy[state])

    def generate_egreedy_episode(self):
        """
        Generate an episode using the epsilon-greedy behavior policy. Will not go longer than self.max_episode_size
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use self.egreedy_selection() above as a helper function
        - use the behavior e-greedy policy attribute aleady calculated (do not update policy here!)
        
        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        #Initialize variables 
        print(self.Q)
        state = self.env.reset()
        state = self.env.get_state()
        counter = 0
        path = []
        #Begin going through episode
        while counter < self.max_episode_size:
            state, action = self.egreedy_selection(state)
            reward = self.env.take_action(int(action))
            path.append((state, action, reward))
            state = self.env.get_state()
            state = tuple(self.env.get_state())
            counter += 1
            if self.env.is_terminal_state():
                break

        return path

        
    
    def generate_greedy_episode(self):
        """
        Generate an episode using the greedy target policy. Will not go longer than self.max_episode_size
        Note: this function is not used during learning, its only for evaluating the target policy
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use the greedy policy attribute aleady calculated (do not update policy here!)

        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        #Initialize variables 
        state = self.env.reset()
        state = tuple(self.env.get_state())
        counter = 0
        path = []

        #Begin going through episode
        while counter < self.max_episode_size:
            action = np.argmax(self.greedy_policy[state])
            reward = self.env.take_action(int(action))
            path.append((state, action, reward))
            state = tuple(self.env.get_state())
            counter += 1
            print((state, action, reward))
            if self.env.is_terminal_state():
                break
        return path
    
    def update_offpolicy(self, episode):
        """
        Update the Q-values using every visit weighted importance sampling. 
        See Figure 5.9, p. 134 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        #Initialize G, W values
        G = 0.0
        W = 1.0

        #Follow Sutton and Barton's pseudocode
        for t in range(len(episode) - 1, 0, -1):
            state, action, reward = episode[t]
            G = self.gamma*G + reward
            if (state, action) not in self.C:
                self.C[(state, action)] = W
            self.C[(state, action)] += W
            self.Q[(state, action)] + self.Q[(state, action)] + W/(self.C[(state, action)])*(G - self.Q[(state, action)])
            # self.greedy_policy[state] = np.argmax(self.Q[state])

            W *= max(self.greedy_policy[state])/max(self.egreedy_policy[state])
            if W == 0:
                break

    
    def update_onpolicy(self, episode):
        """
        Update the Q-values using first visit epsilon-greedy. 
        See Figure 5.6, p. 127 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        visited = []
        returns = defaultdict(np.array)
        for step in episode:
            state, action, reward = step
            state_action_pair = (state, action)
            if state_action_pair not in visited:
                visited.append(state_action_pair)
                G = reward
                returns[state_action_pair].append(G)
                self.Q[state] = sum(returns[state_action_pair])/len(returns[state_action_pair])
        for step in episode:
            a_star = np.argmax(self.Q[state, action])
            for action in range(self.env.n_actions):
                if a_star == action:
                    self.egreedy_policy[state] = 1 - self.epsilon + self.epsilon/abs(self.env.n_actions)
                else:
                    self.egreedy_policy[state] = self.epsilon/abs(self.env.n_actions)




    def train_offpolicy(self, num_episodes):
        """
        Train the agent over a specified number of episodes.
        
        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        for _ in range(num_episodes):
            episode = self.generate_egreedy_episode()
            self.update_offpolicy(episode)

   


    def get_greedy_policy(self):
        """
        Retrieve the learned target policy in the form of an action index per state
        
        Returns:
            dict: The learned target policy.
        """
        policy = {}
        for state, actions in self.Q.items():
            policy[state] = np.argmax(actions)
        return policy
    