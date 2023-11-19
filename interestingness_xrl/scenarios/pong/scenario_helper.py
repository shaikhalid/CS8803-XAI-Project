import matplotlib.pyplot as plt
import numpy as np
import pygame
import pandas as pd
from interestingness_xrl.learning import get_discretized_index
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, TIME_STEPS_VAR, ANY_FEATURE_IDX
from interestingness_xrl.util import print_line
import time
import os
from os.path import join

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
               ANY_FEATURE_IDX: 'any'}

# ELEM_LABELS = {EMPTY_IDX: 'empty',
#                BALL_IDX: 'ball',
#                COMPUTER_PADDLE_IDX: 'computer_paddle',
#                BACKSTOP_IDX: 'backstop',
#                OUT_OF_BOUNDS_IDX: 'out-bounds',
#                ANY_FEATURE_IDX: 'any'}
DIRS = ['up', 'down']
SHORT_DIRS = ['U', 'D']

MAX_POINTS = 21
MIN_Y_POS = 14
MAX_Y_POS = 76

CPU_SCORE = 'cpu_score'
PLAYER_SCORE = 'player_score'
CPU_PADDLE_Y = 'cpu_paddle_y'
PLAYER_PADDLE_Y = 'player_paddle_y'
BALL_X = 'ball_x'
BALL_Y = 'ball_y'



def clean_console():
    """
    Cleans the system's console using the 'cls' or 'clear' command.
    :return:
    """
    os.system('cls' if os.name == 'nt' else 'clear')


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
        # Assuming the agent's location is represented by its paddle's position in the observation
        self.get_coordinate_of_elements(obs)['player_pos']

    def get_cell_coordinates(self, col, row):
        return col * self.config.cell_size[0], MIN_Y_POS + row * self.config.cell_size[1]
        
    def get_coordinate_of_elements(self, obs):
        elem_pos ={ 'player_pos': None, 'ball_pos': None, 'computer_pos': None}
        for row1 in range(14, 73, 5):

            if np.sum(obs[row1:row1+5, 72:75] > 87) >= 2:
                # Store the center pixel as player paddle coordinate
                elem_pos['player_pos'] =  73, 83 - (row1+2)
                break

        for row2 in range(14, 73, 5):
            if np.sum(obs[row2:row2+5, 8:11] > 87) >= 2:
                # Store the center pixel as player paddle coordinate
                elem_pos['computer_pos'] =  9, 83 - (row2+2)
                break
        
        #look for the ball in the empty space between the paddles and the walls
        # top starts from 14th index and bottom ends at 76th index
        # left starts from 11th index and right ends at 71nd index
        for row3 in range(14, 75, 3):
            for col3 in range(11, 70, 3): #69+3 = 71 we take till 70th if we write 71
                if np.sum(obs[row3:row3+3, col3: col3+3] > 87) >= 1:
                    elem_pos['ball_pos'] =  col3+1, 83 - (row3+1)
                    break
        #print(elem_pos)
        return elem_pos

        
        
    def get_features_from_observation(self, obs, feats_nbins=[5, 5]):
        # Take the last frame of the stack for now
        obs = obs[0]

        
              
                # Get the paddle's y-coordinate, idk if I need this for now
        #paddle_y, paddle_x = self.get_paddle_coordinate(obs)
        obs_vec = [EMPTY_IDX, EMPTY_IDX]

        #get the postion of the ball
        elem_coords =  self.get_coordinate_of_elements(obs)
        #print(elem_coords)
        #check if none of the elements are none
        if(elem_coords['ball_pos'] == None):
            with open('obs.txt', 'w') as file:
            # Iterate over the array and write each sub-array to the file on a new line
                for sub_array in obs:
                    # Iterate through each 'row' of the sub-array
                
                    row_str = ' '.join(map(str, sub_array))
                    file.write(row_str + '\n')
        if None not in elem_coords.values():
           
            #get the postion of the ball
            ball_x, ball_y = elem_coords['ball_pos']
            #get the postion of the player paddle
            player_x, player_y,  = elem_coords['player_pos']
            #get the postion of the computer paddle
            computer_x, computer_y= elem_coords['computer_pos']

            #calculate the slope of ball wrt to its previous position
            if self.previous_ball_pos is not None\
                and self.previous_ball_pos != elem_coords['ball_pos']\
                and ball_x > self.previous_ball_pos[0]:
                if ball_x - player_x!= 0 :
                    #calculate the slope between the ball and the player paddle considering ball as the origin
                    theta1 = (ball_y - player_y) / (ball_x - player_x)
                else: 
                    theta1 = 0
                if ball_x - self.previous_ball_pos[0] != 0:
                    #calculate the slope between the ball and its previous position
                    theta2 = (ball_y - self.previous_ball_pos[1]) / (ball_x - self.previous_ball_pos[0])
                else:
                    theta2 = 0

                #print(theta1, theta2)
                #print(ball_y, ball_x, player_y, player_x)
                #if theta1>0 and theta2<0 then set the obs_vec corresponding to down 1
                if theta1 == 0:
                    if theta2 > 0:
                        obs_vec[0] = BALL_IDX
                    if theta2 < 0:
                        obs_vec[1] = BALL_IDX
                if theta2 == 0: # the ball is coming straight
                    if theta1 > 0:
                        obs_vec[1] = BALL_IDX
                    if theta1 < 0:
                        obs_vec[0] = BALL_IDX
                if theta1 > 0 and theta2 < 0:
                    obs_vec[1] = BALL_IDX
                    #print("down 1")
                if theta1 < 0 and theta2 > 0:
                    obs_vec[0] = BALL_IDX
                    #print("up 1")
                if theta1 > 0 and theta2 > 0:
                    if abs(theta1) - abs(theta2) < 0:
                        obs_vec[0] = BALL_IDX
                        #print("up 2")
                    elif abs(theta1) - abs(theta2) > 0 :
                        obs_vec[1] = BALL_IDX
                        #print("down 2")
                if theta1 < 0 and theta2 < 0:
                    if abs(theta1) - abs(theta2) < 0:
                        obs_vec[1] = BALL_IDX
                        #print("down 3")
                    elif abs(theta1) - abs(theta2) > 0 :
                        obs_vec[0] = BALL_IDX
                        #print("up 3")
            self.previous_ball_pos = (ball_x, ball_y)
        #time.sleep(0.2)
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
        return np.linalg.norm(obs1 - obs2)
        # Return a dissimilarity measure between the given observations
        # Placeholder implementation, adjust as needed
        # the difference between the observations (4, ) should be scaled between 0 and 1
    
        

    def get_features_labels(self, obs_vec, short=False):
        # Return a label for the given feature based on its index and value
        feat_labels = [''] * len(obs_vec)
        for i in range(len(DIRS)):
            feat_labels[i] = '{}: {}'.format(SHORT_DIRS[i] if short else DIRS[i], self.get_feature_label(i, obs_vec[i]))
        return feat_labels

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
        #saves table
        print(data)
        print(header)
        df = pd.DataFrame(data, columns=header)
        print(out_file)
        df.to_csv(out_file, delimiter, index=False)

    def get_features_image(self, obs_vec):
        # Convert the given observation into an image representation
        # Placeholder implementation, adjust as needed
        image_size = self.config.cell_size
        surface = pygame.Surface(image_size)
        # Draw features on the surface based on obs_vec
        # ...
        return surface
    
    def update_stats(self, e, t, obs, n_obs, s, a, r, ns, game_state=None):
        super().update_stats(e, t, obs, n_obs, s, a, r, ns)
        
        self.stats_collector.add_sample(CPU_SCORE, e, game_state['cpu_score'])
        self.stats_collector.add_sample(PLAYER_SCORE, e, game_state['player_score'])
        self.stats_collector.add_sample(CPU_PADDLE_Y, e, game_state['cpu_paddle_y'])
        self.stats_collector.add_sample(PLAYER_PADDLE_Y, e, game_state['player_paddle_y'])
        self.stats_collector.add_sample(BALL_X, e, game_state['ball_x'])
        self.stats_collector.add_sample(BALL_Y, e, game_state['ball_y'])
    
    def update_stats_episode(self, e, path=None):
        if (e + 1) % (self.config.num_episodes / 100) == 0:
            clean_console()
            print('Episode {} ({:.0f}%)...'.format(e + 1, ((e + 1) / self.config.num_episodes) * 100))
            #self._print_stats(e, PRINT_SCREEN_VAR_NAMES)

    def save_stats(self, path, clear=True, img_format='pdf'):
        super().save_stats(path, clear)

        # collects and prints final stats to file
        e = self.config.num_episodes
        with open(join(path, 'results.log'), 'w') as file:
            print_line('\nStats (avg. of {} episodes):'.format(e), file)
            var_names = list(self.stats_collector.all_variables())
            var_names.sort()
            self._print_stats(e, var_names, file)

      

           

       


    @staticmethod
    def _load_img(base_dir, img_file):
        return pygame.image.load(join(base_dir, '{}.png'.format(img_file)))