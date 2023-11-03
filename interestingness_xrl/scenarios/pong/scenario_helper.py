import numpy as np
import pygame
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, TIME_STEPS_VAR, ANY_FEATURE_IDX

from interestingness_xrl.util import print_line

    # ball: The ball that the players are trying to hit.
    # player_paddle: The paddle that the player controls.
    # computer_paddle: The paddle that the computer controls.
    # wall: The walls that the ball bounces off of.
    # backstop: The backstop that prevents the ball from going out of bounds.
    # out_of_bounds: The area outside of the playing field.
EMPTY_IDX = 0
BALL_IDX = 1
COMPUTER_PADDLE_IDX = 2
BACKSTOP_IDX = 3
OUT_OF_BOUNDS_IDX = 4

ELEM_LABELS = {EMPTY_IDX: 'empty',
               BALL_IDX: 'ball',
               COMPUTER_PADDLE_IDX: 'computer_paddle',
               BACKSTOP_IDX: 'backstop',
               OUT_OF_BOUNDS_IDX: 'out-bounds',
               ANY_FEATURE_IDX: 'any'}
DIRS = ['up', 'down']
SHORT_DIRS = ['U', 'D']

class PongHelper(ScenarioHelper):
    """
    Represents a set of helper methods for learning and analysis of the Pong game environment.
    """

    def __init__(self, config, img_format='png'):
        super().__init__(config, )
        self.img_format = img_format
        #self.sound = sound

    

    # def register_gym_environment(self, env_id='pong-env-v0', display_screen=False, fps=30, show_score_bar=False):
    #     # Assuming you have a custom Pong environment to register with OpenAI's gym
    #     import gym
    #     from gym.envs.registration import register

    #     register(
    #         id=env_id,
    #         entry_point='path_to_pong_module:PongEnv',  # Replace with the actual path to your Pong environment class
    #     )
    #     return gym.make(env_id)
    def get_features_from_observation(self, obs, agent_x=-1, agent_y=-1):
        # Convert the observation matrix into a set of discretized state features
        # For simplicity, we'll flatten the observation and treat each cell as a feature
        return obs.flatten()

    def get_agent_cell_location(self, obs):
        print(">>>>>obs")
        print(obs)
        # Assuming the agent's location is represented by its paddle's position in the observation
        paddle_y_position = np.where(obs == '1')[0][0]  # Replace 'PADDLE_VALUE' with the actual value representing the paddle in the observation
        return paddle_y_position

    def get_cell_coordinates(self, col, row):
        return col * self.config.cell_size[0], MIN_Y_POS + row * self.config.cell_size[1]
    def get_features_from_observation(self, obs, agent_x=-1, agent_y=-1):
        """
        Transforms the given observation of the Pong environment into a set of discretized state features.
        :param np.ndarray obs: the observation matrix / game state provided to the agent.
        :param int agent_x: the X location of the agent's paddle in the environment. If -1, it has to be collected from the observation.
        :param int agent_y: the Y location of the agent's paddle in the environment. If -1, it has to be collected from the observation.
        :return array: an array containing the discretized state features.
        """

        # Pong specific feature extraction
        player_paddle_pos = obs[0]  # Assuming the player's paddle position is the first element
        opponent_paddle_pos = obs[1]  # Assuming the opponent's paddle position is the second element
        ball_x_pos = obs[2]  # Assuming the ball's x position is the third element
        ball_y_pos = obs[3]  # Assuming the ball's y position is the fourth element
        ball_x_velocity = obs[4]  # Assuming the ball's x velocity is the fifth element
        ball_y_velocity = obs[5]  # Assuming the ball's y velocity is the sixth element

        # Discretize the features if necessary, for example:
        # - Paddle positions could be discretized into 'top', 'middle', 'bottom'.
        # - Ball position could be discretized into different areas of the screen.
        # - Ball velocity could be discretized into 'moving left', 'stationary', 'moving right' and similar for y velocity.

        # For simplicity, let's assume the observation is already discretized and just map them to the feature vector
        obs_vec = [
            player_paddle_pos,  # Discretized player paddle position
            opponent_paddle_pos,  # Discretized opponent paddle position
            ball_x_pos,  # Discretized ball x position
            ball_y_pos,  # Discretized ball y position
            ball_x_velocity,  # Discretized ball x velocity
            ball_y_velocity  # Discretized ball y velocity
        ]

        return obs_vec

    def get_features_bins(self):
        #ignoring any feature, wall, computer paddel and player paddle itself, since they cannot be next to the main paddle
        return [len(ELEM_LABELS) - 1] * len(DIRS)

    def get_terminal_state(self):
        # Return a generic terminal state for the Pong environment
        # This could be when a player scores a point or when the game ends
        return 10  # Placeholder value, adjust as needed
    
    def is_terminal_state(self, obs, rwd, done):
        return False
    
    def is_win_state(self, obs, rwd):
        # if the agent received frog arrived / new level reward, then its a restart
        return False

    def get_observation_dissimilarity(self, obs1, obs2):
        # returns difference of points
        return min(1., abs(obs1[2] - obs2[2]) / MAX_POINTS)

    def get_feature_label(self, obs_feat_idx, obs_feat_val):
        # Return a label for the given feature based on its index and value
        feature_labels = {
            0: "Empty",
            1: "Ball",
            2: "Paddle"
        }
        return feature_labels.get(obs_feat_val, "Unknown")

    def get_features_labels(self, obs_vec, short=False):
        # Return a list of labels for the given state features
        return [self.get_feature_label(idx, val) for idx, val in enumerate(obs_vec)]

    def print_features(self, obs_vec, columns=False):
        # Print a description of the given state features
        labels = self.get_features_labels(obs_vec)
        if columns:
            for label in labels:
                print_line(label)
        else:
            print_line(", ".join(labels))

    def get_transaction(self, obs_vec, short=False):
        # Convert the given set of discretized features into an item-set-like transaction
        return self.get_features_labels(obs_vec, short)

    def act_reactive(self, s, rng):
        # Choose an action based on a known (not learned) strategy for the Pong environment
        # For simplicity, we'll randomly choose an action
        return rng.choice(["UP", "DOWN", "STAY"])

    def get_known_goal_states(self):
        # Return the known goal states for the Pong environment
        # This could be states where a player is about to score
        return []  # Placeholder, adjust as needed

    def get_known_feature_action_assocs(self):
        # Return known associations between action execution and feature states
        # This could be strategies like "if ball is above paddle, move UP"
        return []  # Placeholder, adjust as needed

    def save_state_features(self, out_file, delimiter=','):
        # Save a description of all states in terms of features to a CSV file
        # Placeholder implementation, adjust as needed
        feats_nbins = self.get_features_bins()
        print(feats_nbins)
        num_elems = len(ELEM_LABELS) - 1
        num_states = self.config.num_states
        data = [None] * num_states
    
        for u in range(num_elems):
            for d in range(num_elems):
                # gets discretized index
                obs_vec = [u, d]
                state = get_discretized_index(obs_vec, feats_nbins)

                # puts element names in correct place in table
                data[state] = [state,
                                self.get_feature_label(2, u),
                                self.get_feature_label(3, d)]

        header = [''] * (1+len(DIRS))
        header[0] = 'state'
        for i in range(len(DIRS)):
            header[i + 1] = 'element {}'.format(DIRS[i])
        # saves table
        df = pd.DataFrame(data, columns=header)
        df.to_csv(out_file, delimiter, index=False)

    def get_features_image(self, obs_vec):
        # Convert the given observation into an image representation
        # Placeholder implementation, adjust as needed
        image_size = self.config.cell_size
        surface = pygame.Surface(image_size)
        # Draw features on the surface based on obs_vec
        # ...
        return surface


    @staticmethod
    def _load_img(base_dir, img_file):
        return pygame.image.load(join(base_dir, '{}.png'.format(img_file)))