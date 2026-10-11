import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s,a] += self.alpha*(target - self.Q[s,a])

    def selectionner_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_idx])

    def decroissance_epsilon(self, factor=0.999, min_eps=0.01):
        self.epsilon = max(min_eps, self.epsilon*factor)
        

class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, epsilon=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def update(self,s,a,r,s_next,a_next,done):
        target = r if done else r + self.gamma * self.Q[s_next, a_next]
        self.Q[s,a] += self.alpha*(target - self.Q[s,a])

    def selectionner_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_idx])

    def decroissance_epsilon(self, factor=0.99, min_eps=0.01):
        self.epsilon = max(min_eps, self.epsilon*factor)
