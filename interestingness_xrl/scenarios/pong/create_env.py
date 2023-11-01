#from advertorch.attacks import *
from atari_wrapper import wrap_deepmind
import copy
import torch
#from drl_attacks.uniform_attack import uniform_attack_collector
#from utils import A2CPPONetAdapter

def make_atari_env_watch(env_name):
    return wrap_deepmind(env_name, frame_stack=4,
                         episode_life=False, clip_rewards=False)

# define Pong Atari environment
env = make_atari_env_watch("PongNoFrameskip-v4")
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.env.action_space.shape or env.env.action_space.n

device = 'cuda' if torch.cuda.is_available() else 'cpu'

