import torch
import torch.nn as nn
from torch.distributions import Normal

import os
import numpy as np
from pathlib import Path
from itertools import chain


class POLICY_NETWORK(nn.Module):
    def __init__(self):
        super(POLICY_NETWORK, self).__init__()

        self.feature_map = nn.Sequential(
            nn.Linear(97, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

        self.mean = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 22),
        )

        self.log_std = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 22),
        )

    def forward(self, x):
        feature_map = self.feature_map(x)

        mean = self.mean(feature_map)
        log_std = torch.clamp(self.log_std(feature_map), -20, -2)

        normal = Normal(mean, log_std.exp())
        action = torch.tanh(normal.rsample())

        return action


class Agent(object):
    def __init__(self):
        self.network = POLICY_NETWORK().float().to("cpu")

        self.LoadModel()

    def LoadModel(self):
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        network_path = current_dir / "110060017_hw4_data"
        network_weight = torch.load(network_path)
        self.network.load_state_dict(network_weight["model"])

    def act(self, observation):
        observation = torch.tensor(self.ProcessObservation(observation), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = torch.clamp(self.network(observation), 0, 1).numpy()[0]
        return action

    def ProcessObservation(self, observation):
        # Flatten the musculoskeletal model
        musculoskeletal_model = []
        for item in chain(observation["pelvis"].values(), observation["r_leg"].values(), observation["l_leg"].values()):
            if isinstance(item, list):
                for subitem in item:
                    musculoskeletal_model.append(subitem)
            elif isinstance(item, dict):
                for subitem in item.values():
                    musculoskeletal_model.append(subitem)
            else:
                musculoskeletal_model.append(item)

        # Transform the observation into a numpy array
        observation = np.array(musculoskeletal_model)

        return observation
