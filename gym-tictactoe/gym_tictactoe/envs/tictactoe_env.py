import copy
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

class TicTacToeEnv(Env):
    def __init__(self, size= 3):
        self.size = size
        self.observation_space = Box(low= -1, high= 1, shape= (self.size, self.size), dtype= np.int64)
        self.action_space = Discrete(self.size ** 2)
        self.reward_range = (-1, 1)

        self._player = 1
        self._state_size = (self.size, self.size, 1)
        self._action_to_index_map = self._get_action_to_index_map()
        self._history = []
        self._terminal = False
        self._winner = 0

    def get_observation(self, player):
        if player == 1:
            return self._obs
        elif player == -1:
            return -self._obs
    
    def get_actions(self):
        return [i for i in range(self.action_space.n) if self._obs[self._action_to_index_map[i]] == 0]
    
    def get_result(self, player):
        return self._winner * player
    
    def _get_action_to_index_map(self):
        action_index_map = {}
        for i in range(self.action_space.n):
            action_index_map[i] = (i // self.size, i % self.size)
        return action_index_map
    
    def _is_valid_action(self, action):
        if self._obs[self._action_to_index_map[action]] == 0:
            return True
        return False
    
    def _row_winner(self, obs):
        for row in obs:
            row_sum = sum(row)
            if abs(row_sum) == self.size:
                return row_sum / self.size
        return 0
    
    def _col_winner(self, obs):
        return self._row_winner(obs.T)
    
    def _main_diag_winner(self, obs):
        main_diag_sum = np.trace(obs)
        if abs(main_diag_sum) == self.size:
            return main_diag_sum / self.size
        return 0
    
    def _reverse_main_diag_winner(self, obs):
        return self._main_diag_winner(obs[::-1])
    
    def _is_game_over(self):
        if len(self._history) == self.action_space.n:
            self._winner = 0
            return True
        
        winner = self._row_winner(self._obs)
        if winner != 0:
            self._winner = winner
            return True
        
        winner = self._main_diag_winner(self._obs)
        if winner != 0:
            self._winner = winner
            return True
        
        winner = self._reverse_main_diag_winner(self._obs)
        if winner != 0:
            self._winner = winner
            return True
        
        return False
        
    def _get_info(self):
        return {"history": self._history, "player": self._player, "winner": self._winner}
    
    def step(self, action):
        if not self._is_valid_action(action):
            print('Error: invalid action, please try again')
            return self._obs, self._winner, self._terminal, False, self._get_info()
        else:
            self._obs[self._action_to_index_map[action]] = self._player
            self._history.append(action)
            self._terminal = self._is_game_over()
            if not self._terminal:

                self._player *= -1
            
            return self._obs, self._winner, self._terminal, False, self._get_info()
        
    def reset(self, seed= None, options= None):
        super().reset(seed= seed)

        self._obs = np.zeros((self.size, self.size), dtype= np.int64)
        self._history
        self._terminal = False
        self._winner = 0
        self._player = 1

        return self._obs
    
    def clone(self):
        return copy.deepcopy(self)
    
    def render(self):
        observation = self.get_observation(player= 1)
        # render_game = ''
        # for i in range(len(observation)):
        #     row = ''
        #     for j in range(len(observation[i])):
        #         if observation[i][j] == 0:
        #             row += '.'
        #         elif observation[i][j] == 1:
        #             row += 'X'
        #         elif observation[i][j] == -1:
        #             row += 'O'
        #         else:
        #             row += '-'
        #     row += render_game
        #     render_game += row
        # print(render_game)       
        print(observation)
    