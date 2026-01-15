import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from env import MahjongGBEnv
from feature import FeatureAgent
from model import ResnetModel
from torch import nn
from collections import defaultdict
import random
import os
import logging
import matplotlib.pyplot as plt
import warnings

# 计算每个时间步的回报和优势
def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    advantages = []
    G = 0
    A = 0

    for i in reversed(range(len(rewards))):
        if dones[i]:
            G = 0
            A = 0

        G = rewards[i] + gamma * G
        returns.insert(0, G)

        if i < len(values) - 1:
            next_value = values[i + 1]
        else:
            next_value = 0

        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        A = delta + gamma * lam * A * (1 - dones[i])
        advantages.insert(0, A)

    return returns, advantages


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])

    return returns, advantages


# PPO 更新过程
def ppo_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    # 建议：优势标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {
                'is_training': True,
                'obs': {
                    'observation': state_batch,
                    'action_mask': action_mask_batch
                }
            }

            action_logits, value = policy_network(input_dict)

            # 如果 action_mask 是 0/1，强烈建议在 softmax 前做 mask
            action_logits = action_logits.masked_fill(action_mask_batch == 0, -1e9)

            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            # ================= PPO 核心部分 =================
            # 1. ratio = π(a|s) / π_old(a|s)
            ratio = torch.exp(log_prob - old_log_prob_batch)

            # 2. clip
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_batch

            # 3. policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # 4. value loss (MSE)
            value = value.squeeze(-1)
            value_loss = torch.mean((return_batch - value) ** 2)
            # =================================================

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)


def reinforce_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
                     clip_param=None):
    """
    这里是标准 REINFORCE + baseline 版本
    实际上不需要 clip_param 和 old_log_probs，但为了接口统一保留
    """
    returns = returns.detach()

    dataset = TensorDataset(states, actions, returns, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, action_mask_batch = batch

            input_dict = {
                'is_training': True,
                'obs': {
                    'observation': state_batch,
                    'action_mask': action_mask_batch
                }
            }

            action_logits, value = policy_network(input_dict)

            # 同样建议 mask
            action_logits = action_logits.masked_fill(action_mask_batch == 0, -1e9)

            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            # policy loss: -E[ log π(a|s) * R ]
            policy_loss = -(log_prob * return_batch).mean()

            # baseline 用 value，value loss 仍然是 MSE
            value = value.squeeze(-1)
            value_loss = torch.mean((return_batch - value) ** 2)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)

