'''Neural network model'''
from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config

# 代理器状态，包含了观测值、动作维度、上一个动作、上一个奖励和隐藏状态
@dataclass
class AgentState:
    obs: torch.Tensor # 相当于next_obs
    action_dim: int # 动作维度
    last_action: torch.Tensor = field(init=False) # 得到next_obs的动作执行的动作
    last_reward: torch.Tensor = torch.zeros((1, 1), dtype=torch.float32)
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __post_init__(self):
        self.last_action = torch.zeros((1, self.action_dim), dtype=torch.float32)
    
    def update(self, obs, last_action, last_reward, hidden):
        '''
        obs: 最新观察值
        last_action: 到该obs所执行的动作
        last_reward: 到该obs所获得的奖励
        hidden: LSTM的隐藏状态
        '''
        self.obs = torch.from_numpy(obs).unsqueeze(0)
        # 将动作转换为one-hot编码
        self.last_action = torch.FloatTensor([[1 if i == last_action else 0 for i in range(self.action_dim)]])
        self.last_reward = torch.FloatTensor([[last_reward]])
        self.hidden_state = hidden


class Network(nn.Module):
    def __init__(self, action_dim, obs_shape=config.obs_shape, hidden_dim=config.hidden_dim):
        super().__init__()

        # 84 x 84 input

        self.action_dim = action_dim
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim

        self.max_forward_steps = config.forward_steps

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(True),
        )

        self.recurrent = nn.LSTM(512+self.action_dim+1, self.hidden_dim, batch_first=True)

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state: AgentState):
        
        # 归一化 0～1
        # 提取观察特征
        latent = self.feature(state.obs / 255)

        # 当前观察的特征、上一个动作和上一个奖励拼接
        recurrent_input = torch.cat((latent, state.last_action, state.last_reward), dim=1)

        # 将拼接后的状态和隐藏状态输入到LSTM中
        _, recurrent_output = self.recurrent(recurrent_input, state.hidden_state)

        # 这里获取的是隐藏层的状态
        hidden = recurrent_output[0]

        # 利用隐藏层的状态计算 Q 值
        adv = self.advantage(hidden) # 优势函数 A(s,a)，表示在当前状态下选择特定动作相对于平均的优势
        val = self.value(hidden) # 状态值函数 V(s)，表示在当前状态下的价值
        # 这行代码在计算 Dueling DQN 架构中的 Q 值
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # adv.mean(1, keepdim=True) 计算所有动作的优势平均值减去平均优势是为了提高数值稳定性，使优势函数的期望为 0
        q_value = val + adv - adv.mean(1, keepdim=True)

        # 返回预计的Q值和隐藏状态
        return q_value, recurrent_output

    def calculate_q_(self, obs, last_action, last_reward, hidden_state, burn_in_steps, learning_steps, forward_steps):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)
        last_reward = last_reward.view(-1, 1)
        # 观察特征提取
        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps + forward_steps

        recurrent_input = torch.cat((latent, last_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)

        '''
        # 1. 原始数据维度
        latent: (batch_size * max_seq_len, 512)
        last_action: (batch_size * max_seq_len, action_dim)
        last_reward: (batch_size * max_seq_len, 1)

        # 2. 拼接后的维度
        recurrent_input: (batch_size * max_seq_len, 512+action_dim+1)

        # 3. 重塑为序列形式
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        # 维度变为: (batch_size, max_seq_len, 512+action_dim+1)

        # 4. 打包序列
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, 
                                            batch_first=True, 
                                            enforce_sorted=False)
        '''
        # 这个函数用于处理变长序列，将填充的序列压缩成紧凑形式，去除冗余的填充部分
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, batch_first=True, enforce_sorted=False)

        self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        # 将压缩的序列解压回原来的形状
        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        seq_start_idx = burn_in_steps + self.max_forward_steps
        forward_pad_steps = torch.minimum(self.max_forward_steps - forward_steps, learning_steps)

        hidden = []
        for hidden_seq, start_idx, end_idx, padding_length in zip(recurrent_output, seq_start_idx, seq_len, forward_pad_steps):
            hidden.append(hidden_seq[start_idx:end_idx])
            if padding_length > 0:
                hidden.append(hidden_seq[end_idx-1:end_idx].repeat(padding_length, 1))

        hidden = torch.cat(hidden)

        assert hidden.size(0) == torch.sum(learning_steps)

        adv = self.advantage(hidden)
        val = self.value(hidden)
        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value


    def calculate_q(self, obs, last_action, last_reward, hidden_state, burn_in_steps, learning_steps):
        # obs shape: (batch_size, seq_len, obs_shape)
        batch_size, max_seq_len, *_ = obs.size()

        obs = obs.reshape(-1, *self.obs_shape)
        last_action = last_action.view(-1, self.action_dim)
        last_reward = last_reward.view(-1, 1)

        latent = self.feature(obs)

        seq_len = burn_in_steps + learning_steps

        recurrent_input = torch.cat((latent, last_action, last_reward), dim=1)
        recurrent_input = recurrent_input.view(batch_size, max_seq_len, -1)
        recurrent_input = pack_padded_sequence(recurrent_input, seq_len, batch_first=True, enforce_sorted=False)

        # self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(recurrent_input, hidden_state)

        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)

        hidden = torch.cat([output[burn_in:burn_in+learning] for output, burn_in, learning in zip(recurrent_output, burn_in_steps, learning_steps)], dim=0)

        adv = self.advantage(hidden)
        val = self.value(hidden)

        q_value = val + adv - adv.mean(1, keepdim=True)

        return q_value