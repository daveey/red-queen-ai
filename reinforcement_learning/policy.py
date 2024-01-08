import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.emulation
import pufferlib.models

from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class Random(pufferlib.models.Policy):
    '''A random policy that resets weights on every call'''
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(1, n) for n in env.single_action_space.nvec]
        )

    def encode_observations(self, flat_observations):
        return torch.randn((flat_observations.shape[0], 1)).to(flat_observations.device), None

    def decode_actions(self, flat_hidden, lookup):
        torch.nn.init.xavier_uniform_(flat_hidden)
        actions = [dec(flat_hidden) for dec in self.decoders]
        return actions, None

    def critic(self, hidden):
        return torch.zeros((hidden.shape[0], 1)).to(hidden.device)


class Baseline(pufferlib.models.Policy):
    def __init__(self, env, input_size=256, hidden_size=256):
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.proj_fc = torch.nn.Linear(2 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_observations):
        env_outputs = pufferlib.emulation.unpack_batched_obs(flat_observations,
            self.flat_observation_space, self.flat_observation_structure)
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )
        obs = torch.cat([tile, my_agent], dim=-1)
        obs = self.proj_fc(obs)

        return obs, (
            player_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, flat_hidden, lookup):
        actions = self.action_decoder(flat_hidden, lookup)
        value = self.value_head(flat_hidden)
        return actions, value


class TileEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tile_offset = torch.tensor([i * 256 for i in range(3)])
        self.embedding = torch.nn.Embedding(3 * 256, 32)

        self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

    def forward(self, tile):
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7
        tile = self.embedding(
            tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
        )

        agents, tiles, features, embed = tile.shape
        tile = (
            tile.view(agents, tiles, features * embed)
            .transpose(1, 2)
            .view(agents, features * embed, 15, 15)
        )

        tile = F.relu(self.tile_conv_1(tile))
        tile = F.relu(self.tile_conv_2(tile))
        tile = tile.contiguous().view(agents, -1)
        tile = F.relu(self.tile_fc(tile))

        return tile


class PlayerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.entity_dim = 31
        self.player_offset = torch.tensor([i * 256 for i in range(self.entity_dim)])
        self.embedding = torch.nn.Embedding(self.entity_dim * 256, 32)

        self.agent_fc = torch.nn.Linear(self.entity_dim * 32, hidden_size)
        self.my_agent_fc = torch.nn.Linear(self.entity_dim * 32, input_size)

    def forward(self, agents, my_id):
        # Pull out rows corresponding to the agent
        agent_ids = agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        agent_embeddings = self.embedding(
            agents.long().clip(0, 255) + self.player_offset.to(agents.device)
        )
        batch, agent, attrs, embed = agent_embeddings.shape

        # Embed each feature separately
        agent_embeddings = agent_embeddings.view(batch, agent, attrs * embed)
        my_agent_embeddings = agent_embeddings[
            torch.arange(agents.shape[0]), row_indices
        ]

        # Project to input of recurrent size
        agent_embeddings = self.agent_fc(agent_embeddings)
        my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
        my_agent_embeddings = F.relu(my_agent_embeddings)

        return agent_embeddings, my_agent_embeddings

class ActionDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "attack_style": torch.nn.Linear(hidden_size, 3),
                "attack_target": torch.nn.Linear(hidden_size, hidden_size),
                "comm_token": torch.nn.Linear(hidden_size, 50),
                "move": torch.nn.Linear(hidden_size, 5),
            }
        )

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)

        return hidden

    def forward(self, hidden, lookup):
        (
            player_embeddings,
            action_targets,
        ) = lookup

        embeddings = {
            "attack_target": player_embeddings,
        }

        action_targets = {
            "attack_style": action_targets["Attack"]["Style"],
            "attack_target": action_targets["Attack"]["Target"],
            "comm_token": action_targets["Comm"]["Token"],
            "move": action_targets["Move"]["Direction"],
        }

        actions = []
        for key, layer in self.layers.items():
            mask = action_targets[key]
            embs = embeddings.get(key)
            if embs is not None and embs.shape[1] != mask.shape[1]:
                b, _, f = embs.shape
                zeros = torch.zeros([b, 1, f], dtype=embs.dtype, device=embs.device)
                embs = torch.cat([embs, zeros], dim=1)

            action = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action)

        return actions
