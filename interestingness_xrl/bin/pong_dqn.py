import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from interestingness_xrl.bin.net.discrete_net import DQN
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from interestingness_xrl.bin.custom_collector import CustomCollector
from interestingness_xrl.scenarios.pong.atari_wrapper import wrap_deepmind, InverseReward
from interestingness_xrl.scenarios import DEFAULT_CONFIG, create_helper, AgentType, get_agent_output_dir, create_agent
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios.pong.configurations import *
from os import makedirs
from os.path import join, exists

DEF_AGENT_TYPE = AgentType.Learning
DEF_TRIAL = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps_test', type=float, default=0.005)
    parser.add_argument('--eps_train', type=float, default=1.)
    parser.add_argument('--eps_train_final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--target_update_freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--collect_per_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--training_num', type=int, default=16)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    parser.add_argument('--invert_reward', default=False, action='store_true',
                        help="rew'=-rew")
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config', help='path to config file')
    parser.add_argument('-t', '--trial', help='trial number to run', type=int, default=DEF_TRIAL)
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    return parser.parse_args()


def make_atari_env(args):
    environment = wrap_deepmind(args.task, frame_stack=args.frames_stack)
    if args.invert_reward:
        environment = InverseReward(environment)
    return environment


def make_atari_env_watch(args):
    environment = wrap_deepmind(args.task, frame_stack=args.frames_stack,
                                episode_life=False, clip_rewards=False)
    if args.invert_reward:
        environment = InverseReward(environment)
    return environment


def test_dqn(args=get_args()):
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    agent_t = args.agent
    if agent_t == AgentType.Testing:

        # tries to load config from provided results dir path
        results_dir = args.results if args.results is not None else \
            get_agent_output_dir(PONG_CONFIG, AgentType.Learning)
        config_file = join(results_dir, 'config.json')
        if not exists(results_dir) or not exists(config_file):
            raise ValueError('Could not load configuration from: {}.'.format(config_file))
        config = EnvironmentConfiguration.load_json(config_file)

        # if testing, we want to force a seed different than training (diff. test environments)
        config.seed += 1
    else:
        # tries to load env config from provided file path
        config_file = args.config
        config = PONG_CONFIG if config_file is None or not exists(config_file) \
            else EnvironmentConfiguration.load_json(config_file)
        
    output_dir = args.output if args.output is not None else get_agent_output_dir(config, agent_t, args.trial)
    if not exists(output_dir):
        makedirs(output_dir)

    # creates env helper
    helper = create_helper(config)
    print(config)
    config.save_json(join(output_dir, 'config.json'))
    helper.save_state_features(join(output_dir, 'state_features.csv'))

    # should be N_FRAMES x H x W
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: make_atari_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = DQN(*args.state_shape,
              args.action_shape, args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(net, optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)
    buffer = ReplayBuffer(args.buffer_size, ignore_obs_next=True)
    # collector
    print("Creating collector ...")
    train_collector = CustomCollector(helper, policy, train_envs, buffer)
    test_collector = CustomCollector(helper, policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        elif 'Pong' in args.task:
            return x >= 20

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                  (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        writer.add_scalar('train/eps', eps, global_step=env_step)
        print("set eps =", policy.eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Testing agent ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=args.render)
        pprint.pprint(result)

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * 4)
    
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer, test_in_train=False)

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())