import os
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from dataclasses import asdict


@dataclass
class ArgsSac:
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
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "highway-v0"
    """the environment id of the task"""
    total_timesteps: int = 160000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 150000 #int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


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

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod() + np.prod(env.action_space.n),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape).prod())
        act_dim = env.action_space.n  # number of discrete actions, e.g. 5

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_logits = nn.Linear(256, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)   # unnormalized log-probabilities
        return logits

    def get_action(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                     # sample an action index
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, logits


class SACAgent:
    def __init__(self, env, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.env = env
        self.args = args
        self.global_step = 0
        

        self.actor = Actor(env).to(self.device)
        self.qf1 = SoftQNetwork(env).to(self.device)
        self.qf2 = SoftQNetwork(env).to(self.device)
        self.qf1_target = SoftQNetwork(env).to(self.device)
        self.qf2_target = SoftQNetwork(env).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr)

        if args.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

        self.rb = ReplayBuffer(
            args.buffer_size,
            env.observation_space,
            env.action_space,
            self.device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )

    @torch.no_grad()
    def predict(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32).reshape(1, -1).to(self.device)
        action, _, mean = self.actor.get_action(obs.unsqueeze(0) if obs.ndim == 1 else obs)
        return mean.cpu().numpy() if deterministic else action.cpu().numpy()

    def add_transition(self, obs, next_obs, action, reward, done, info):
        self.rb.add(obs, next_obs, action, reward, done, info)

    def update(self):
        if self.global_step < self.args.learning_starts:
            return

        data = self.rb.sample(self.args.batch_size)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(data.next_observations)
            q1_next = self.qf1_target(data.next_observations, next_actions)
            q2_next = self.qf2_target(data.next_observations, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * min_q_next.view(-1)

        # Critic update
        q1 = self.qf1(data.observations, data.actions).view(-1)
        q2 = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(q1, target_q)
        qf2_loss = F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.q_optimizer.step()

        # Delayed actor update
        if self.global_step % self.args.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(data.observations)
            q1_pi = self.qf1(data.observations, pi)
            q2_pi = self.qf2(data.observations, pi)
            actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Entropy tuning
            if self.args.autotune:
                alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

        # Target networks
        if self.global_step % self.args.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        self.global_step += 1


    def save(self, model_path):
        torch.save(self.qf1.state_dict(), f"{model_path}_qf1")
        args_dict = asdict(self.args)
        torch.save(self.qf2.state_dict(), f"{model_path}_qf2")
        args_dict = asdict(self.args)

        json_path = os.path.join(os.path.dirname(model_path), "hyperparameters.json")
        with open(json_path, "w") as f:
            json.dump(args_dict, f, indent=4)

        print(f"model saved to {model_path}")
        print(f"Hyperparameters saved to {json_path}")

    def load(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))

        self.loaded = True
        print(f"model loaded from {model_path}")
