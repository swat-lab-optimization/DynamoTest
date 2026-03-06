# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass, asdict
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from agents.abstract_agent import AbstractAgent

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 100000)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "dmhum"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "highway-fast-v0"
    """the id of the environment"""
    total_timesteps: int =  160000#80000#160000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 150000 #150000
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2 #0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 5000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk




class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

class DQNAgentCLRL(AbstractAgent):
    def __init__(self, envs, args: Args, writer=None):
        self.name = "DQN"
        self.envs = envs
        self.args = args
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.start_time = time.time()

        self.q_network = QNetwork(envs).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.target_network = QNetwork(envs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(args.buffer_size, envs.observation_space, envs.action_space, self.device, handle_timeout_termination=False)

        self.global_step = 0
        self.loaded = False


    @staticmethod
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def predict(self, obs, first=False):
        epsilon = DQNAgentCLRL.linear_schedule(self.args.start_e, self.args.end_e, self.args.exploration_fraction * self.args.total_timesteps, self.global_step)
        if random.random() < epsilon:
            actions = np.array([self.envs.action_space.sample()])
        else:
            if self.loaded:
                with torch.no_grad():
                    q_values = self.q_network(torch.Tensor(obs).reshape(1, -1).to(self.device))
            else:
                q_values = self.q_network(torch.Tensor(obs).reshape(1, -1).to(self.device))
            
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        return actions

    def update(self, obs, actions, next_obs, rewards, infos, terminations, truncations):
        real_next_obs = next_obs.copy()

        if truncations:
            real_next_obs = obs.copy()

        self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # ALGO LOGIC: training.
        if self.global_step > self.args.learning_starts:
            if self.global_step % self.args.train_frequency == 0:
                data = self.rb.sample(self.args.batch_size)
                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations.reshape(data.next_observations.shape[0], -1)).max(dim=1)
                    td_target = data.rewards.flatten() + self.args.gamma * target_max * (1 - data.dones.flatten())
                old_val = self.q_network(data.observations.reshape(data.observations.shape[0], -1)).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if self.global_step % 100 == 0 and self.writer is not None:
                    self.writer.add_scalar("losses/td_loss", loss, self.global_step)
                    self.writer.add_scalar("losses/q_values", old_val.mean().item(), self.global_step)
                    print("SPS:", int(self.global_step / (time.time() - self.start_time)))
                    self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update target network
            if self.global_step % self.args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_network_param.data.copy_(
                        self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                    )
        
        self.global_step += 1

    def save(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)
        args_dict = asdict(self.args)

        json_path = os.path.join(os.path.dirname(model_path), "hyperparameters.json")
        with open(json_path, "w") as f:
            json.dump(args_dict, f, indent=4)

        print(f"model saved to {model_path}")
        print(f"Hyperparameters saved to {json_path}")

    def load(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.args.start_e = 0.05
        self.args.end_e = 0.01
        self.loaded = True
        print(f"model loaded from {model_path}")