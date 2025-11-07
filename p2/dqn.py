# -*- coding:utf-8 -*-
import argparse
open = __builtins__.open  # 修复 gym video_recorder NameError
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.6,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.3,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--tau", type=float, default=0.005,
        help="soft update coefficient for target network")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """comments: 
    Define the class of a Deep-Q-Network
    Use an NN to describe Q
    Q(s, a)=NN(s)
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments: 
    Used for decaying epsilon
    And here we use a linear-deacying method
    while limited the minimum as end_e
    Use t(time) for decreasing
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """comments: 
    Init everything we need
    including random-seeds and the device selecting
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    

    """comments: 
    build our environments
    """
    envs = make_env(args.env_id, args.seed)
    # 创建模型和视频保存目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./videos', exist_ok=True)

    """comments: 
    Init the NN
    Send to the device we select
    Use Adam-optimizer
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """comments: 
    build our replaybuffer
    Used for Experience Replay
    """
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """comments: 
    Start Training!
    """
    obs = envs.reset()
    epsilon = args.start_e
    for global_step in range(args.total_timesteps):
        
        """comments: 
        Calculate the epsilon
        """
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """comments: 
        Use Epsilon-Greedy Algorithm to generate our action
        """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """comments: 
        Step forward according to the action we choose
        And then record the return and length into the tensorboard
        """
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        # =====Reward Shaping========
        # shaped_rewards = rewards

        # x = next_obs[0]
        # y = next_obs[1]
        # vx = next_obs[2]
        # vy = next_obs[3]
        # angle = next_obs[4]
        # angulat_vel = next_obs[5]

        # if 0 < y < 1.0:
        #     height_reward = 0.3 * (1.0 - y)
        #     shaped_rewards += height_reward

        # speed = np.sqrt(vx * vx + vy * vy)
        # if speed > 1.0:
        #     speed_reward = -0.2 * (speed - 1.0)
        #     shaped_rewards += speed_reward

        # if abs(angle) < 0.3:
        #     angle_reward = 0.2 * (1.0 - abs(angle / 0.3))
        #     shaped_rewards += angle_reward
        
        # if abs(x) < 0.5:
        #     position_reward = 0.1 * (1.0 - abs(x) / 0.5)
        #     shaped_rewards += position_reward

        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """comments: 
        Put our experience into the ReplayBuffer
        """
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """comments: 
        Refresh our obs
        """
        obs = next_obs if not dones else envs.reset()

        # 每50000步导出一个示例视频
        if (global_step > 0) and (global_step % 50000 == 0):
            # 生成视频环境
            video_env = gym.make(args.env_id)
            video_env = gym.wrappers.RecordVideo(
                video_env,
                video_folder='./videos',
                name_prefix=f"step_{global_step}",
                episode_trigger=lambda episode_id: True
            )
            test_obs = video_env.reset()
            done = False
            try:
                while not done:
                    # 用当前模型选择动作（评估模式）
                    state = torch.tensor(test_obs, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        q_values = q_network(state)
                        action = torch.argmax(q_values).item()
                    test_obs, reward, done, info = video_env.step(action)
            finally:
                video_env.close()
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """comments: 
            Sampling experince from the buffer
            """
            data = rb.sample(args.batch_size)
            
            """comments: 
            Calculate the TD_Target, the Q for now and the MSE-Loss
            old_val used to mean into the q_value
            loss used to mean the td_loss
            """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """comments: 
            Record the q and loss into TensorBoard
            """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """comments: 
            Backward-propagation to calculate the gradient
            The use the gradient-descending
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """comments: 
            Refresh out target_network regularly
            """
            # if global_step % args.target_network_frequency == 0:
            #     target_network.load_state_dict(q_network.state_dict())

            # Add Soft Update
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
            # ==========================================
    
    # 训练完成后保存模型
    torch.save(q_network.state_dict(), './models/dqn_final.pth')

    """close the env and tensorboard logger"""
    envs.close()
    writer.close()