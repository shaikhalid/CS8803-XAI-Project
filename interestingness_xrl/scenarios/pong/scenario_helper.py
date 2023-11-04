import matplotlib.pyplot as plt
import numpy as np
import pygame
import pandas as pd
from interestingness_xrl.learning import get_discretized_index
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, TIME_STEPS_VAR, ANY_FEATURE_IDX
from interestingness_xrl.util import print_line
import time

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
        self.previous_ball_pos = None
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
        print(obs)
        # Assuming the agent's location is represented by its paddle's position in the observation
        paddle_y_position = np.where(obs == '1')[0][0]  # Replace 'PADDLE_VALUE' with the actual value representing the paddle in the observation
        return paddle_y_position

    def get_cell_coordinates(self, col, row):
        return col * self.config.cell_size[0], MIN_Y_POS + row * self.config.cell_size[1]
    
 
        
    def get_coordinate_of_elements(self, obs):
        elem_pos ={ 'player_pos': None, 'ball_pos': None, 'computer_pos': None}
        for row1 in range(14, 73, 5):

            if np.sum(obs[row1:row1+5, 72:75] > 87) >= 5:
                # Store the center pixel as player paddle coordinate
                elem_pos['player_pos'] = row1+2, 73

        for row2 in range(14, 73, 5):
            if np.sum(obs[row2:row2+5, 8:11] > 87) >= 5:
                # Store the center pixel as player paddle coordinate
                elem_pos['computer_pos'] = row2+2, 9

        for row3 in range(14, 75, 3):
            for col3 in range(1, obs.shape[1]-2, 3):
                # Check if at least one grid has exactly 4 pixel values greater than 87
                # and the border elements of the 4x4 matrix should only consist of 87
                if np.sum(obs[row3:row3+3, col3:col3+3] > 87) >= 3 and \
                   np.all(obs[row3:row3+5, col3-2] == 87) and \
                   np.all(obs[row3:row3+5, col3+4] == 87) and \
                   np.all(obs[row3-2, col3:col3+5] == 87) and \
                   np.all(obs[row3+4, col3:col3+5] == 87):
                    # Store the center pixel as ball coordinate
                    elem_pos['ball_pos'] = row3+1, col3+1
                    break
        print(elem_pos)
        return elem_pos

        
        
    def get_features_from_observation(self, obs, feats_nbins=[5, 5]):
        # Take the last frame of the stack for now
        obs = obs[-1]

        with open('obs.txt', 'w') as file:
            # Iterate over the array and write each sub-array to the file on a new line
            for sub_array in obs:
                # Iterate through each 'row' of the sub-array
               
                row_str = ' '.join(map(str, sub_array))
                file.write(row_str + '\n')
              
                # Get the paddle's y-coordinate, idk if I need this for now
        #paddle_y, paddle_x = self.get_paddle_coordinate(obs)
        obs_vec = [EMPTY_IDX, EMPTY_IDX]

        #get the postion of the ball
        self.get_coordinate_of_elements(obs)['ball_pos']
        # print("coor", ball_y, ball_x)
        time.sleep(0.5)

        # # calculate the slope of the line between the prev postion of the ball and the current position of the ball
        # if self.previous_ball_pos is not None and self.previous_ball_pos != (ball_y, ball_x):
        #     if ball_x - self.previous_ball_pos[1] != 0:
        #         slope = (ball_y - self.previous_ball_pos[0]) / (ball_x - self.previous_ball_pos[1])
        #         #if slope if postive make the obs_vec curresponding to up 1
        #         if slope > 0:
        #             obs_vec[0] = BALL_IDX
        #         #if slope is negative make the obs_vec curresponding to down 1
        #         elif slope < 0:
        #             obs_vec[1] = BALL_IDX
        #     else:
        #         # handle division by zero
        #         if ball_y > self.previous_ball_pos[0]:
        #             obs_vec[0] = BALL_IDX
        #         elif ball_y < self.previous_ball_pos[0]:
        #             obs_vec[1] = BALL_IDX

        # self.previous_ball_pos = ball_y, ball_x
        
   
        
        return obs_vec


    #need to add win state
    def get_state_from_observation(self, obs, rwd, done):
        #return self.win_state if self.is_win_state(obs, rwd) else super().get_state_from_observation(obs, rwd, done)
        return super().get_state_from_observation(obs, rwd, done)

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

    def get_feature_label(self, obs_feat_idx, obs_feat_val):
        return ELEM_LABELS[obs_feat_val]

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
                                self.get_feature_label(0, u),
                                self.get_feature_label(1, d)]

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