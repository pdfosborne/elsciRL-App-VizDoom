import os
import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gymnasium as gym
import itertools


class Engine:
    def __init__(self, local_setup_info:dict):
        self.config_path = os.path.join(os.getcwd(), local_setup_info['config_path'])
        self.frame_skip = local_setup_info['frame_skip']
        self.render_mode = local_setup_info['render_mode']
        
        
        self.game = vzd.DoomGame()
        self.game.load_config(self.config_path)
        self.state_history = []
        self.action_history = []


        screen_format = self.game.get_screen_format()
        if (
            screen_format != vzd.ScreenFormat.RGB24
            and screen_format != vzd.ScreenFormat.GRAY8
        ):
            warnings.warn(
                f"Detected screen format {screen_format.name}. Only RGB24 and GRAY8 are supported in the Gymnasium"
                f" wrapper. Forcing RGB24."
            )
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        self.channels = 3
        if screen_format == vzd.ScreenFormat.GRAY8:
            self.channels = 1

        self.state = None
        self.clock = None
        self.window_surface = None
        self.isopen = True
    
        self.depth = self.game.is_depth_buffer_enabled()
        self.labels = self.game.is_labels_buffer_enabled()
        self.automap = self.game.is_automap_buffer_enabled()

        # parse buttons defined by config file
        self.__parse_available_buttons()

        # check for valid max_buttons_pressed
        max_buttons_pressed = local_setup_info['max_buttons_pressed']
        if max_buttons_pressed > self.num_binary_buttons > 0:
            warnings.warn(
                f"max_buttons_pressed={max_buttons_pressed} "
                f"> number of binary buttons defined={self.num_binary_buttons}. "
                f"Clipping max_buttons_pressed to {self.num_binary_buttons}."
            )
            max_buttons_pressed = self.num_binary_buttons
        elif max_buttons_pressed < 0:
            raise RuntimeError(
                f"max_buttons_pressed={max_buttons_pressed} < 0. Should be >= 0. "
            )

        # specify action space(s)
        self.max_buttons_pressed = max_buttons_pressed
        self.action_space = self.__get_action_space()

        # specify observation space(s)
        self.observation_space = self.__get_observation_space()

        self.game.init()

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

    def __parse_binary_buttons(self, env_action, agent_action):
        if self.num_binary_buttons != 0:
            if self.num_delta_buttons != 0:
                agent_action = agent_action["binary"]

            if np.issubdtype(type(agent_action), np.integer):
                agent_action = self.button_map[agent_action]

            # binary actions offset by number of delta buttons
            env_action[self.num_delta_buttons :] = agent_action

    def __parse_delta_buttons(self, env_action, agent_action):
        if self.num_delta_buttons != 0:
            if self.num_binary_buttons != 0:
                agent_action = agent_action["continuous"]

            # delta buttons have a direct mapping since they're reorganized to be prior to any binary buttons
            env_action[0 : self.num_delta_buttons] = agent_action

    def __build_env_action(self, agent_action):
        # encode users action as environment action
        env_action = np.array(
            [0 for _ in range(self.num_delta_buttons + self.num_binary_buttons)],
            dtype=np.float32,
        )
        self.__parse_delta_buttons(env_action, agent_action)
        self.__parse_binary_buttons(env_action, agent_action)
        return env_action
    
    def __parse_available_buttons(self):
        """
        Parses the currently available game buttons,
        reorganizes all delta buttons to be prior to any binary buttons
        sets ``num_delta_buttons``, ``num_binary_buttons``
        """
        delta_buttons = []
        binary_buttons = []
        for button in self.game.get_available_buttons():
            if vzd.is_delta_button(button) and button not in delta_buttons:
                delta_buttons.append(button)
            else:
                binary_buttons.append(button)
        # force all delta buttons to be first before any binary buttons
        self.game.set_available_buttons(delta_buttons + binary_buttons)
        self.num_delta_buttons = len(delta_buttons)
        self.num_binary_buttons = len(binary_buttons)
        if delta_buttons == binary_buttons == 0:
            raise RuntimeError(
                "No game buttons defined. Must specify game buttons using `available_buttons` in the "
                "config file."
            )

    def __get_binary_action_space(self):
        """
        Return binary action space: either ``Discrete(n)``/``MultiDiscrete([2] * num_binary_buttons)``
        """
        if self.max_buttons_pressed == 0:
            button_space = gym.spaces.MultiDiscrete(
                [
                    2,
                ]
                * self.num_binary_buttons
            )
        else:
            self.button_map = [
                np.array(list(action))
                for action in itertools.product((0, 1), repeat=self.num_binary_buttons)
                if (self.max_buttons_pressed >= sum(action) >= 0)
            ]
            button_space = gym.spaces.Discrete(len(self.button_map))
        return button_space

    def __get_continuous_action_space(self):
        """
        Returns continuous action space: Box(float32.min, float32.max, (num_delta_buttons,), float32)
        """
        return gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            (self.num_delta_buttons,),
            dtype=np.float32,
        )

    def __get_action_space(self):
        """
        Returns action space:
            if both binary and delta buttons defined in the config file, action space will be:
              ``Dict("binary": MultiDiscrete|Discrete, "continuous", Box)``
            else:
              action space will be only one of the following ``MultiDiscrete``|``Discrete``|``Box``
        """
        if self.num_delta_buttons == 0:
            return self.__get_binary_action_space()
        elif self.num_binary_buttons == 0:
            return self.__get_continuous_action_space()
        else:
            return gym.spaces.Dict(
                {
                    "binary": self.__get_binary_action_space(),
                    "continuous": self.__get_continuous_action_space(),
                }
            )

    def __get_observation_space(self):
        """
        Returns observation space: Dict with Box entry for each activated buffer:
          "screen", "depth", "labels", "automap", "gamevariables"
        """
        spaces = {
            "screen": gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.channels,
                ),
                dtype=np.uint8,
            )
        }

        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.automap:
            spaces["automap"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    # "automap" buffer uses same number of channels
                    # as the main screen buffer,
                    self.channels,
                ),
                dtype=np.uint8,
            )

        self.num_game_variables = self.game.get_available_game_variables_size()
        if self.num_game_variables > 0:
            spaces["gamevariables"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (self.num_game_variables,),
                dtype=np.float32,
            )

        return gym.spaces.Dict(spaces)

    def reset(self, start_obs=None):
        self.game.new_episode()
        self.state_history = []
        self.action_history = []
        self.state = self.game.get_state()
        
        return self.__collect_observations()

    def step(self, state, action):
        env_action = self.__build_env_action(action)
        reward = self.game.make_action(env_action, self.frame_skip)
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
