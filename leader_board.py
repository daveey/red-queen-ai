
import pufferlib
import pufferlib.emulation


class StatPostprocessor(pufferlib.emulation.Postprocessor):
    """Postprocessing actions and metrics of Neural MMO.
       Process wandb/leader board stats, and save replays.
    """
    def __init__(self, env, is_multiagent, agent_id, eval_mode=False):
        super().__init__(env, is_multiagent, agent_id=agent_id)
        self.eval_mode = eval_mode
        self._reset_episode_stats()

    def reset(self, observation):
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.epoch_return = 0
        self.epoch_length = 0

        # for agent results
        self._damage_received = 0
        self._damage_inflicted = 0

    def _update_stats(self, agent):
        self._damage_received += agent.history.damage_received
        self._damage_inflicted += agent.history.damage_inflicted

    def observation(self, observation):
        return observation

    def action(self, action):
        return action

    def reward_done_info(self, reward, done, info):
        """Update stats + info and save replays."""

        # Count and store unique event counts for easier use
        # log = self.env.realm.event_log.get_data(agents=[self.agent_id])

        if not done:
            self.epoch_length += 1
            self.epoch_return += reward
            return reward, done, info

        if 'stats' not in info:
            info['stats'] = {}

        agent = self.env.realm.players.dead_this_tick.get(
            self.agent_id, self.env.realm.players.get(self.agent_id)
        )
        assert agent is not None
        self._update_stats(agent)

        info['return'] = self.epoch_return
        info['length'] = self.epoch_length

        return reward, done, info
