__author__ = 'Khalid Shaikh'
__email__ = 'shaikhalidwork@gmail.com'

from collections import OrderedDict
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration

GAME_GYM_ID = 'PongNoFrameskip-v4'

class PongConfiguration(EnvironmentConfiguration):
    """
    A class used to store configurations of tests / learning simulations on the Pong game.
    """

    def __init__(self, name, actions, rewards, gym_env_id, max_steps_per_episode, num_episodes, num_recorded_videos, 
                 seed, max_temp, min_temp, discount, learn_rate, initial_q_value):
        """
        Creates a new configuration with the given parameters.
        :param str name: the name of the configuration.
        :param OrderedDict actions: the actions available for the agent.
        :param dict rewards: the reward function.
        :param str gym_env_id: the name identifier for the gym environment.
        :param int max_steps_per_episode: the maximum number of steps in one episode.
        :param int num_episodes: the number of episodes used to train/test the agent.
        :param int num_recorded_videos: the number of videos to record during the test episodes.
        :param int seed: the seed used for the random number generator used by the agent.
        :param float max_temp: the maximum temperature of the Soft-max action-selection strategy (start of training).
        :param float min_temp: the minimum temperature of the Soft-max action-selection strategy (end of training).
        :param float discount: the discount factor in [0, 1] (how important are the future rewards?).
        :param float learn_rate: the agent's learning rate.
        :param float initial_q_value: the value used to initialize the q-function.
        """
        # Assuming a simplified state representation for Pong (this can be adjusted)
        # element = ball, computer_paddle, backstop, out-of-bounds
        # directions = up and down
        num_states = 5**2  

        super().__init__(name, num_states, actions, rewards, gym_env_id, max_steps_per_episode, num_episodes,
                         num_recorded_videos, seed, max_temp, min_temp, discount, learn_rate, initial_q_value)

PONG_CONFIG = PongConfiguration(
    name='pong',
    actions=OrderedDict([
        ('noop', 0),
        ('fire', 1),
        ('right', 2),
        ('left', 3),
        ('rightfire', 4),
        ('leftfire', 5)
    ]),
    rewards={
        'score_point': 5000.,  # Reward for scoring a point
        'lose_point': -200.    # Penalty for losing a point
    },
    gym_env_id=GAME_GYM_ID,
    max_steps_per_episode=300,
    num_episodes=2000,
    num_recorded_videos=10,
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.3,
    initial_q_value=0.
)
