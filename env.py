import numpy as np


class GridWorld:
    def __init__(self, size):
        self.size = size
        self.n_states = size * size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {(2,2),(2,3),(3,2),(5,6),(6,5),(6,6),
                          (4,4), (6,3), (3,6), (12,2), (10,10), 
                          (3,16), (16,16), (16,18), (18,19)
                          }
        self.actions = [(-1,0), (1,0),  (0,-1), (0,1)]
        #               haut    bas    gauche  droite
        self.n_actions = 4

    def reset(self):
        return self.start
    
    def step(self, state, action_idx):
        # On récupérer l'action/mouvement correspondant à l'action cnoisie
        dr, dc = self.actions[action_idx]
        next_state = (state[0] + dr, state[1] + dc)
        # Récompense/pénalité pour les bords
        if not (0 <= next_state[0] < self.size and 0 <= next_state[1] < self.size):
            return state, -1.0, False
        # Récompense/pénalité pour les obstacles
        if next_state in self.obstacles:
            return state, -1.0, False
        # Récompense pour avoir atteint le but/l'objectif
        if next_state == self.goal:
            return next_state, 10.0, True
        # Récompense normale
        return next_state, -0.1, False
    
    def state_to_idx(self, state):
        return state[0] * self.size + state[1]
