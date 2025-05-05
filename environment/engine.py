import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt


class VizDoomEngine:
    def __init__(self, local_setup_info:dict):
        self.config_path = local_setup_info.get('config_path', 'basic.cfg')
        self.game = vzd.DoomGame()
        self.game.load_config(self.config_path)
        self.game.init()
        self.state_history = []
        self.action_history = []

    def reset(self, start_obs=None):
        self.game.new_episode()
        self.state_history = []
        self.action_history = []
        state = self.game.get_state()
        return state

    def step(self, state, action):
        reward = self.game.make_action(action)
        terminated = self.game.is_episode_finished()
        info = {}
        next_state = None if terminated else self.game.get_state()
        self.state_history.append(state)
        self.action_history.append(action)
        return next_state, reward, terminated, info

    def legal_move_generator(self, state=None):
        return list(range(self.game.get_available_buttons_size()))

    def render(self, state=None):
        frame = state.screen_buffer if state else self.game.get_state().screen_buffer
        fig = plt.figure()
        plt.imshow(np.transpose(frame, (1, 2, 0)))
        plt.axis('off')
        return fig

    def close(self):
        self.game.close()