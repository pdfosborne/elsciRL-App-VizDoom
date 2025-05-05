import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt


class Engine:
    def __init__(self, local_setup_info:dict):
        self.config_path = local_setup_info.get('config_path', 'basic.cfg')
        self.game = vzd.DoomGame()
        self.game.load_config(self.config_path)
        self.game.init()
        self.state_history = []
        self.action_history = []
    
    def __collect_observations(self):
        observation = {}
        if self.state is not None:
            observation["screen"] = self.state.screen_buffer
            if self.channels == 1:
                observation["screen"] = self.state.screen_buffer[..., None]
            if self.depth:
                observation["depth"] = self.state.depth_buffer[..., None]
            if self.labels:
                observation["labels"] = self.state.labels_buffer[..., None]
            if self.automap:
                observation["automap"] = self.state.automap_buffer
                if self.channels == 1:
                    observation["automap"] = self.state.automap_buffer[..., None]
            if self.num_game_variables > 0:
                observation["gamevariables"] = self.state.game_variables.astype(
                    np.float32
                )
        else:
            # there is no state in the terminal step, so a zero observation is returned instead
            for space_key, space_item in self.observation_space.spaces.items():
                observation[space_key] = np.zeros(
                    space_item.shape, dtype=space_item.dtype
                )

        return observation
    
    def reset(self, start_obs=None):
        self.game.new_episode()
        self.state_history = []
        self.action_history = []
        self.state = self.game.get_state()
        
        return self.__collect_observations()

    def step(self, state, action):
        reward = self.game.make_action(action)
        terminated = self.game.is_episode_finished()
        info = {}
        self.state = None if terminated else self.game.get_state()
        self.state_history.append(self.state)
        self.action_history.append(action)
        return self.__collect_observations(), reward, terminated, info

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
