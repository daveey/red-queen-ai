from argparse import Namespace
import math

import nmmo
import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor, calculate_entropy

class Config(nmmo.config.Default):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.MAP_FORCE_GENERATION = False
        self.PLAYER_N = args.num_agents
        self.HORIZON = args.max_episode_length
        self.MAP_N = args.num_maps
        self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
        self.MAP_CENTER = args.map_size

        self.COMMUNICATION_SYSTEM_ENABLED = False


class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id, eval_mode=False):
        super().__init__(env, is_multiagent, agent_id, eval_mode)

    def reset(self, observation):
        '''Called at the start of each episode'''
        super().reset(observation)

    @property
    def observation_space(self):
        '''If you modify the shape of features, you need to specify the new obs space'''
        return super().observation_space

    """
    def observation(self, obs):
        '''Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        '''
        return obs

    def action(self, action):
        '''Called before actions are passed from the model to the environment'''
        return action
    """

    def reward_done_info(self, reward, done, info):
        '''Called on reward, done, and info before they are returned from the environment'''

        reward, done, info = super().reward_done_info(reward, done, info)
        return reward, done, info


def make_env_creator(args: Namespace):
    # TODO: Max episode length
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                'eval_mode': args.eval_mode,
            },
        )
        return env
    return env_creator
