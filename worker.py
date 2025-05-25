'''Replay buffer, learner and actor'''
import time
import random
import os
import math
from copy import deepcopy
from typing import List, Tuple
import threading
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from model import Network, AgentState
from environment import create_env
from priority_tree import PriorityTree
import config

############################## Replay Buffer ##############################


@dataclass
class Block:
    # 存储观察值序列
    # shape: (sequence_length, *obs_shape)
    obs: np.array

    # 存储上一个时刻执行的动作的one-hot编码
    # shape: (sequence_length, action_dim)
    last_action: np.array

    # 存储上一个时刻获得的奖励
    # shape: (sequence_length,)
    last_reward: np.array

    # 存储当前时刻执行的动作的标量值
    # shape: (sequence_length,)
    action: np.array

    # 存储n步累积奖励(使用n-step TD)
    # shape: (sequence_length,)
    n_step_reward: np.array

    # 存储n步折扣因子
    # shape: (sequence_length,)
    gamma: np.array

    # 存储LSTM的隐藏状态
    # shape: (num_sequences, 2, hidden_dim)
    hidden: np.array

    # 当前block包含的序列数量
    num_sequences: int

    # 每个序列的burn-in步数 todo
    # shape: (num_sequences,)
    burn_in_steps: np.array

    # 每个序列的实际学习步数 todo
    # shape: (num_sequences,)
    learning_steps: np.array

    # 每个序列的前向展望步数(用于n-step return) todo
    # shape: (num_sequences,)
    forward_steps: np.array


class ReplayBuffer:
    def __init__(self, sample_queue_list, batch_queue, priority_queue, buffer_capacity=config.buffer_capacity, sequence_len=config.block_length,
                alpha=config.prio_exponent, beta=config.importance_sampling_exponent,
                batch_size=config.batch_size):

        self.buffer_capacity = buffer_capacity # 缓冲区的大小
        self.sequence_len = config.learning_steps # todo
        self.num_sequences = buffer_capacity//self.sequence_len # todo
        self.block_len = config.block_length # todo block的长度 block时做什么用的
        self.num_blocks = self.buffer_capacity // self.block_len # todo 计算有多少个block
        self.seq_pre_block = self.block_len // self.sequence_len # todo 每个block中包含的序列数

        self.block_ptr = 0 # todo

        # todo 感觉应该是实现了一个优先级缓冲区
        self.priority_tree = PriorityTree(self.num_sequences, alpha, beta)

        self.batch_size = batch_size

        self.env_steps = 0
        
        self.num_episodes = 0
        self.episode_reward = 0

        self.training_steps = 0
        self.last_training_steps = 0
        self.sum_loss = 0

        self.lock = threading.Lock()

        self.size = 0 # 这个应该是最新的数据采集量
        self.last_size = 0 # 这个应该是上一次的数据采集量

        self.buffer = [None] * self.num_blocks

        # todo 这三个队列是什么意思？
        self.sample_queue_list, self.batch_queue, self.priority_queue = sample_queue_list, batch_queue, priority_queue

    def __len__(self):
        return self.size

    def run(self):
        # todo 这三个进程的作用分别是什么？
        background_thread = threading.Thread(target=self.add_data, daemon=True)
        background_thread.start()

        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()

        background_thread = threading.Thread(target=self.update_data, daemon=True)
        background_thread.start()

        log_interval = config.log_interval

        # 这边看起来只是打印日志，真做事的还是上面三个线程
        while True:
            print(f'buffer size: {self.size}')
            print(f'buffer update speed: {(self.size-self.last_size)/log_interval}/s')
            self.last_size = self.size
            print(f'number of environment steps: {self.env_steps}')
            if self.num_episodes != 0:
                print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
                # print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
                self.episode_reward = 0
                self.num_episodes = 0
            print(f'number of training steps: {self.training_steps}')
            print(f'training speed: {(self.training_steps-self.last_training_steps)/log_interval}/s')
            if self.training_steps != self.last_training_steps:
                print(f'loss: {self.sum_loss/(self.training_steps-self.last_training_steps):.4f}')
                self.last_training_steps = self.training_steps
                self.sum_loss = 0
            self.last_env_steps = self.env_steps
            print()

            if self.training_steps == config.training_steps:
                break
            else:
                time.sleep(log_interval)

    def prepare_data(self):
        # 要等到缓冲区有足够的数据才能开始采样
        while self.size < config.learning_starts:
            time.sleep(1)

        while True:
            # 知道缓冲区满了才开始采样数据
            if not self.batch_queue.full():
                data = self.sample_batch()
                self.batch_queue.put(data)
            else:
                time.sleep(0.1)

    def add_data(self):
        while True:
            for sample_queue in self.sample_queue_list:
                if not sample_queue.empty():
                    data = sample_queue.get_nowait()
                    self.add(*data)

    def update_data(self):

        while True:
            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)
            else:
                time.sleep(0.1)


    def add(self, block: Block, priority: np.array, episode_reward: float):
        '''
        block: 整个block的内容，包含了一个episode的所有数据
        priority: 该block的优先级
        episode_reward: 该episode的总奖励，如果是探索阶段则为0
        '''

        with self.lock:

            idxes = np.arange(self.block_ptr*self.seq_pre_block, (self.block_ptr+1)*self.seq_pre_block, dtype=np.int64)

            # 根据计算的优先级更新对应索引的优先级
            # todo 后续看看这个优先级是怎么计算的
            self.priority_tree.update(idxes, priority)

            if self.buffer[self.block_ptr] is not None:
                self.size -= np.sum(self.buffer[self.block_ptr].learning_steps).item()

            self.size += np.sum(block.learning_steps).item()
            
            # 将数据存储到缓冲区中
            self.buffer[self.block_ptr] = block

            # todo
            self.env_steps += np.sum(block.learning_steps, dtype=np.int32)

            # 更新缓冲区的指针，这里模拟的是一个循环缓冲区的效果
            self.block_ptr = (self.block_ptr+1) % self.num_blocks
            if episode_reward:
                self.episode_reward += episode_reward # 这里应该是记录一个episode的总奖励，应该是用于计算平均奖励值
                self.num_episodes += 1

    def sample_batch(self):
        '''sample one batch of training data'''
        batch_obs, batch_last_action, batch_last_reward, batch_hidden, batch_action, batch_reward, batch_gamma = [], [], [], [], [], [], []
        burn_in_steps, learning_steps, forward_steps = [], [], []

        with self.lock:
            
            # 这里的idxes应该是对应的序列索引
            idxes, is_weights = self.priority_tree.sample(self.batch_size)

            block_idxes = idxes // self.seq_pre_block  # 确定在哪个block，表示每个索引在哪个block的索引
            sequence_idxes = idxes % self.seq_pre_block  # 确定block中的哪个序列 标识每个索引在block中的序列索引


            for block_idx, sequence_idx  in zip(block_idxes, sequence_idxes):

                block = self.buffer[block_idx]

                assert sequence_idx < block.num_sequences, 'index is {} but size is {}'.format(sequence_idx, self.seq_pre_block_buf[block_idx])

                burn_in_step = block.burn_in_steps[sequence_idx]
                learning_step = block.learning_steps[sequence_idx]
                forward_step = block.forward_steps[sequence_idx]
                
                start_idx = block.burn_in_steps[0] + np.sum(block.learning_steps[:sequence_idx])

                obs = block.obs[start_idx-burn_in_step:start_idx+learning_step+forward_step]
                last_action = block.last_action[start_idx-burn_in_step:start_idx+learning_step+forward_step]
                last_reward = block.last_reward[start_idx-burn_in_step:start_idx+learning_step+forward_step]
                obs, last_action, last_reward = torch.from_numpy(obs), torch.from_numpy(last_action), torch.from_numpy(last_reward)
                
                start_idx = np.sum(block.learning_steps[:sequence_idx])
                end_idx = start_idx + block.learning_steps[sequence_idx]
                action = block.action[start_idx:end_idx]
                reward = block.n_step_reward[start_idx:end_idx]
                gamma = block.gamma[start_idx:end_idx]
                hidden = block.hidden[sequence_idx]
                
                batch_obs.append(obs)
                batch_last_action.append(last_action)
                batch_last_reward.append(last_reward)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_gamma.append(gamma)
                batch_hidden.append(hidden)

                burn_in_steps.append(burn_in_step)
                learning_steps.append(learning_step)
                forward_steps.append(forward_step)

            batch_obs = pad_sequence(batch_obs, batch_first=True)
            batch_last_action = pad_sequence(batch_last_action, batch_first=True)
            batch_last_reward = pad_sequence(batch_last_reward, batch_first=True)

            is_weights = np.repeat(is_weights, learning_steps)


            data = (
                batch_obs,
                batch_last_action,
                batch_last_reward,
                torch.from_numpy(np.stack(batch_hidden)).transpose(0, 1),

                torch.from_numpy(np.concatenate(batch_action)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_reward)),
                torch.from_numpy(np.concatenate(batch_gamma)),

                torch.ByteTensor(burn_in_steps),
                torch.ByteTensor(learning_steps),
                torch.ByteTensor(forward_steps),

                idxes,
                torch.from_numpy(is_weights.astype(np.float32)),
                self.block_ptr,

                self.env_steps
            )

        return data

    def update_priorities(self, idxes: np.ndarray, td_errors: np.ndarray, old_ptr: int, loss: float):
        """Update priorities of sampled transitions"""
        """更新优先级，应该是在训练的时候利用损失"""
        with self.lock:

            # discard the idxes that already been replaced by new data in replay buffer during training
            if self.block_ptr > old_ptr:
                # range from [old_ptr, self.seq_ptr)
                mask = (idxes < old_ptr*self.seq_pre_block) | (idxes >= self.block_ptr*self.seq_pre_block)
                idxes = idxes[mask]
                td_errors = td_errors[mask]
            elif self.block_ptr < old_ptr:
                # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.seq_pre_block) & (idxes >= self.block_ptr*self.seq_pre_block)
                idxes = idxes[mask]
                td_errors = td_errors[mask]

            self.priority_tree.update(idxes, td_errors)

        self.training_steps += 1
        self.sum_loss += loss




############################## Learner ##############################

def calculate_mixed_td_errors(td_error, learning_steps):
    # todo 调试看看如何计算的'''
    
    start_idx = 0
    mixed_td_errors = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        mixed_td_errors[i] = 0.9*td_error[start_idx:start_idx+steps].max() + 0.1*td_error[start_idx:start_idx+steps].mean()
        start_idx += steps
    
    return mixed_td_errors

class Learner:
    '''
    这个应该是一个学习器，模型训练的地方
    '''
    def __init__(self, batch_queue, priority_queue, model, grad_norm: int = config.grad_norm,
                lr: float = config.lr, eps:float = config.eps, game_name: str = config.game_name,
                target_net_update_interval: int = config.target_net_update_interval, save_interval: int = config.save_interval):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_net = deepcopy(model) # 模型的拷贝
        self.online_net.to(self.device)
        self.online_net.train()
        self.target_net = deepcopy(self.online_net) # 模型的拷贝 目标模型
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=eps) # 只针对online_net进行优化
        self.loss_fn = nn.MSELoss(reduction='none')
        self.grad_norm = grad_norm # 梯度裁剪的阈值
        self.batch_queue = batch_queue # todo
        self.priority_queue = priority_queue # todo
        self.num_updates = 0
        self.done = False

        self.target_net_update_interval = target_net_update_interval # 目标模型的同步频率
        self.save_interval = save_interval # 模型的保存频率

        self.batched_data = [] # todo

        self.shared_model = model # todo 共享模型，用于更新不同线程之间的权重吗？

        self.game_name = game_name # todo

    def store_weights(self):
        self.shared_model.load_state_dict(self.online_net.state_dict())

    def prepare_data(self):

        while True:
            if not self.batch_queue.empty() and len(self.batched_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batched_data.append(data)
            else:
                time.sleep(0.1)

    def run(self):
        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()
        time.sleep(2)

        start_time = time.time()
        while self.num_updates < config.training_steps:
            
            while not self.batched_data:
                time.sleep(1)
            data = self.batched_data.pop(0)

            batch_obs, batch_last_action, batch_last_reward, batch_hidden, batch_action, batch_n_step_reward, batch_n_step_gamma, burn_in_steps, learning_steps, forward_steps, idxes, is_weights, old_ptr, env_steps = data
            batch_obs, batch_last_action, batch_last_reward = batch_obs.to(self.device), batch_last_action.to(self.device), batch_last_reward.to(self.device)
            batch_hidden, batch_action = batch_hidden.to(self.device), batch_action.to(self.device)
            batch_n_step_reward, batch_n_step_gamma = batch_n_step_reward.to(self.device), batch_n_step_gamma.to(self.device)
            is_weights = is_weights.to(self.device)

            batch_obs, batch_last_action = batch_obs.float(), batch_last_action.float()
            batch_action = batch_action.long()
            burn_in_steps, learning_steps, forward_steps = burn_in_steps, learning_steps, forward_steps

            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            batch_obs = batch_obs / 255

            # double q learning
            with torch.no_grad():
                batch_action_ = self.online_net.calculate_q_(batch_obs, batch_last_action, batch_last_reward, batch_hidden, burn_in_steps, learning_steps, forward_steps).argmax(1).unsqueeze(1)
                batch_q_ = self.target_net.calculate_q_(batch_obs, batch_last_action, batch_last_reward, batch_hidden, burn_in_steps, learning_steps, forward_steps).gather(1, batch_action_).squeeze(1)
            
            target_q = self.value_rescale(batch_n_step_reward + batch_n_step_gamma * self.inverse_value_rescale(batch_q_))
            # target_q = batch_n_step_reward + batch_n_step_gamma * batch_q_

            batch_q = self.online_net.calculate_q(batch_obs, batch_last_action, batch_last_reward, batch_hidden, burn_in_steps, learning_steps).gather(1, batch_action).squeeze(1)
            
            loss = (is_weights * self.loss_fn(batch_q, target_q)).mean()

            
            td_errors = (target_q-batch_q).detach().clone().squeeze().abs().cpu().float().numpy()

            priorities = calculate_mixed_td_errors(td_errors, learning_steps.numpy())

            # automatic mixed precision training
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
            self.optimizer.step()

            self.num_updates += 1

            self.priority_queue.put((idxes, priorities, old_ptr, loss.item()))

            # store new weights in shared memory
            if self.num_updates % 4 == 0:
                self.store_weights()

            # update target net
            if self.num_updates % self.target_net_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            
            # save model 
            if self.num_updates % self.save_interval == 0:
                torch.save((self.online_net.state_dict(), self.num_updates, env_steps, (time.time()-start_time)/60), os.path.join('models', '{}{}.pth'.format(self.game_name, self.num_updates)))

    @staticmethod
    def value_rescale(value, eps=1e-3):
        return value.sign()*((value.abs()+1).sqrt()-1) + eps*value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4*eps*(value.abs()+1+eps)).sqrt() - 1) / (2*eps)
        return value.sign() * (temp.square() - 1)


############################## Actor ##############################

class LocalBuffer:
    '''store transitions of one episode 这个应该是一个局部缓冲区，存储一个episode的所有数据'''
    def __init__(self, action_dim: int, forward_steps: int = config.forward_steps,
                burn_in_steps = config.burn_in_steps, learning_steps: int = config.learning_steps, 
                gamma: float = config.gamma, hidden_dim: int = config.hidden_dim, block_length: int = config.block_length):
        '''
        action_dim: 动作的维度
        hidden_dim: 隐藏层的维度 todo 作用
        '''
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.forward_steps = forward_steps # n步dqn计算时的前向步数
        self.learning_steps = learning_steps
        self.burn_in_steps = burn_in_steps
        self.block_length = block_length
        self.curr_burn_in_steps = 0
        
    def __len__(self):
        return self.size
    
    def reset(self, init_obs: np.ndarray):
        '''
        这个函数的作用 初始化buffer缓冲区
        1. 在动作器每个环境重置时都会调用一次

        init_obs: 初始的观察值
        '''
        self.obs_buffer = [init_obs] # 这里的obs_buffer 对应着相同位置的action表示的是到达该obs所执行的动作
        # np.array([1 if i == 0 else 0 for i in range(self.action_dim) 这里实际时创建一个one-hot向量，这个one-hot变量只的是动作的向量，指向的是none动作
        # 这里指的是初始初始状态时，上一个执行的动作是none，存储的是动作的one-hont编码
        self.last_action_buffer = [np.array([1 if i == 0 else 0 for i in range(self.action_dim)], dtype=bool)]
        # 这里指的是初始状态时，上一个执行的奖励是0
        self.last_reward_buffer = [0]
        # 隐藏状态的初始值是0
        self.hidden_buffer = [np.zeros((2, self.hidden_dim), dtype=np.float32)]
        self.action_buffer = [] # 存储动作的标量值，每次游戏结束时都会清空缓冲区
        self.reward_buffer = [] # 存储的也是环境的奖励 ，每次游戏结束时都会清空缓冲区
        self.qval_buffer = [] # 存储的是动作的q值，每次游戏结束时都会清空缓冲区
        self.curr_burn_in_steps = 0 # todo
        self.size = 0 # 缓冲区的大小
        self.sum_reward = 0 # 存储的是应该是一个生命周期内获取的总奖励 todo
        self.done = False

    def add(self, action: int, reward: float, next_obs: np.ndarray, q_value: np.ndarray, hidden_state: np.ndarray):
        '''
        action: 执行的动作
        reward: 得到的奖励
        next_obs: 得到的状态
        q_value: 得到的Q值
        hidden_state: 模型隐藏层的状态
        '''
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.hidden_buffer.append(hidden_state)
        self.obs_buffer.append(next_obs)
        self.last_action_buffer.append(np.array([1 if i == action else 0 for i in range(self.action_dim)], dtype=bool))
        self.last_reward_buffer.append(reward)
        self.qval_buffer.append(q_value)
        self.sum_reward += reward
        self.size += 1
    
    def finish(self, last_qval: np.ndarray = None) -> Tuple:
        '''
        last_qval: 在游戏结束时会调用这个方法，并且这里值为None/当达到最大步数时，会调用这个方法，并且这里值为为下一个Q值
        '''
        assert self.size <= self.block_length
        # assert len(self.last_action_buffer) == self.curr_burn_in_steps + self.size + 1

        # 这行代码在计算需要多少个序列(sequences)来存储当前缓冲区的所有数据:
        # todo 估计就是训练时时一个连续序列一个序列取训练的 todo
        num_sequences = math.ceil(self.size/self.learning_steps)

        # 最大的前进步数，这里应该就是限制最大的训练步数 todo 作用
        max_forward_steps = min(self.size, self.forward_steps)
        # 感觉有点像n步DQN的计算变量 todo
        n_step_gamma = [self.gamma**self.forward_steps] * (self.size-max_forward_steps)

        # last_qval is none means episode done 
        if last_qval is not None:
            # todo 这里是干嘛
            self.qval_buffer.append(last_qval)
            n_step_gamma.extend([self.gamma**i for i in reversed(range(1, max_forward_steps+1))])
        else:
            self.done = True # 游戏结束标识
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[0])) # 如果游戏结束了，那么最新的Q值则是0
            # todo
            n_step_gamma.extend([0 for _ in range(max_forward_steps)]) # set gamma to 0 so don't need 'done'
        
        # 将N步DQN的gamma转换为矩阵，方便计算
        n_step_gamma = np.array(n_step_gamma, dtype=np.float32)

        obs = np.stack(self.obs_buffer) # 将所有的观察值堆叠成一个数组
        last_action = np.stack(self.last_action_buffer) # 将所有的上一个动作堆叠成一个数组
        last_reward = np.array(self.last_reward_buffer, dtype=np.float32) # 将所有的上一个奖励堆叠成一个数组
        
        '''
        slice(start, stop, step) 是Python的切片操作，有三个参数：

        start: 起始索引
        stop: 结束索引（不包含）
        step: 步长

        这等价于更常见的切片语法：self.hidden_buffer[0:self.size:self.learning_steps]

        所以以下操作是按照固定的步长为每一个训练的序列提取起始的隐藏层状态用于训练
        '''
        hiddens = np.stack(self.hidden_buffer[slice(0, self.size, self.learning_steps)])

        actions = np.array(self.action_buffer, dtype=np.uint8)

        qval_buffer = np.concatenate(self.qval_buffer)
        # 这里的作用应该是填充奖励的长度，可能是为了对齐某种长度
        reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps-1)]
        # 这里是计算n步dqn时，每一个时刻的前向forward_steps所累积的奖励
        '''
        # 衰减因子序列为：
        [0.9^2, 0.9^1, 0.9^0] = [0.81, 0.9, 1.0]

        # 如果 reward_buffer = [1, 2, 3, 4, 0, 0]
        # 卷积计算 n 步累积奖励：
        # 1*0.81 + 2*0.9 + 3*1.0 = 4.71
        # 2*0.81 + 3*0.9 + 4*1.0 = 7.02
        # 3*0.81 + 4*0.9 + 0*1.0 = 6.03
        # 4*0.81 + 0*0.9 + 0*1.0 = 3.24
        '''
        n_step_reward = np.convolve(reward_buffer, 
                                    [self.gamma**(self.forward_steps-1-i) for i in range(self.forward_steps)],
                                    'valid').astype(np.float32)

        '''
        curr_burn_in_steps 在 R2D2 算法中是用来跟踪当前的 burn-in 步数的变量。它的主要作用是：

        维护 LSTM 状态的连续性
        在序列之间保持 LSTM 隐藏状态的连续性
        确保 LSTM 在训练时有正确的上下文信息

        todo 怎么使用？
        '''
        burn_in_steps = np.array([min(i*self.learning_steps+self.curr_burn_in_steps, self.burn_in_steps) for i in range(num_sequences)], dtype=np.uint8)
        # 这里应该是存储实际用于训练的学习步数，如果对于结尾部分需要考虑到不是learning_steps倍数的情况，所以采用的是self.size-i*self.learning_steps
        learning_steps = np.array([min(self.learning_steps, self.size-i*self.learning_steps) for i in range(num_sequences)], dtype=np.uint8)
        # 同样也是类似的道理，这里是n步dqn的向前看的步数，对于结尾部分 todo 调试看
        forward_steps = np.array([min(self.forward_steps, self.size+1-np.sum(learning_steps[:i+1])) for i in range(num_sequences)], dtype=np.uint8)
        assert forward_steps[-1] == 1 and burn_in_steps[0] == self.curr_burn_in_steps
        # assert last_action.shape[0] == self.curr_burn_in_steps + np.sum(learning_steps) + 1

        # R2D2中TD误差和优先级计算的详细解析 todo 调试查看
        # 从qval_buffer中提取未来状态的最大Q值,这里的未来状态应该是越过了训练步长的状态
        max_qval = np.max(qval_buffer[max_forward_steps:self.size+1], axis=1)
        # 使用边缘填充('edge')来处理序列末尾的值
        max_qval = np.pad(max_qval, (0, max_forward_steps-1), 'edge')
        # 选择实际上每一步执行动作的Q值
        target_qval = qval_buffer[np.arange(self.size), actions]

        # 算n步TD误差的绝对值
        # n_step_reward: n步累积奖励
        # n_step_gamma * max_qval: 折扣后的未来最大Q值
        # target_qval: 当前动作的Q值
        # 这个误差用于优先级回放
        # 误差雨大则说明预测的Q值和实际的Q值差距大，优先级回放会更倾向于选择这种误差大的样本进行训练
        td_errors = np.abs(n_step_reward + n_step_gamma * max_qval - target_qval, dtype=np.float32)
        # 计算优先级:
        # 创建优先级数组
        # 使用混合TD误差计算优先级
        # 混合TD误差结合了最大和平均TD误差
        # 这些优先级用于优先级经验回放采样
        priorities = np.zeros(self.block_length//self.learning_steps, dtype=np.float32)
        # 更新优先级
        priorities[:num_sequences] = calculate_mixed_td_errors(td_errors, learning_steps)

        # save burn in information for next block
        # 看起来像保留一部分之前的数据，用于维护LSTM的上下文状态
        self.obs_buffer = self.obs_buffer[-self.burn_in_steps-1:]
        self.last_action_buffer = self.last_action_buffer[-self.burn_in_steps-1:]
        self.last_reward_buffer = self.last_reward_buffer[-self.burn_in_steps-1:]
        self.hidden_buffer = self.hidden_buffer[-self.burn_in_steps-1:]
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.qval_buffer.clear()
        self.curr_burn_in_steps = len(self.obs_buffer)-1
        self.size = 0
        
        # 将本次的数据打包成Block对象，一个block包含一个周期内所有的序列数据
        block = Block(obs, last_action, last_reward, actions, n_step_reward, n_step_gamma, hiddens, num_sequences, burn_in_steps, learning_steps, forward_steps)
        return [block, priorities, self.sum_reward if self.done else None]


class Actor:
    '''
    游戏环境的采集器
    '''
    def __init__(self, epsilon: float, model, sample_queue, obs_shape: np.ndarray = config.obs_shape,
                max_episode_steps: int = config.max_episode_steps, block_length: int = config.block_length):
        '''
        epsilon: epsilon-greedy策略的epsilon值
        model: 共享模型
        sample_queue: 采集器采集到的数据存放的队列
        obs_shape: 观察值的形状
        max_episode_steps: 每个episode的最大步数
        block_length: 每个block的长度 todo 作用
        '''

        # 创建游戏环境
        self.env = create_env(noop_start=True)
        self.action_dim = self.env.action_space.n
        # 这里应该是创建当前动作器的模型，并设置为评估模式
        self.model = Network(self.env.action_space.n)
        self.model.eval()
        # todo
        self.local_buffer = LocalBuffer(self.action_dim)

        self.epsilon = epsilon
        self.shared_model = model
        self.sample_queue = sample_queue # 存储着block，每个block包含一个周期内的所有序列数据
        self.max_episode_steps = max_episode_steps
        self.block_length = block_length

    def run(self):
        
        actor_steps = 0 # 采集器执行的步数，在整个采集器的生命周期内，会不断的增加下去

        while True:

            done = False # 是否结束
            agent_state = self.reset() # 重置环境，这里返回的是AgentState
            episode_steps = 0 # 生命周期内执行的步数

            # 如果游戏没有结束或者没有达到最大步数则继续执行
            while not done and episode_steps < self.max_episode_steps:
                
                # 利用模型预测动作的Q值和隐藏状态
                with torch.no_grad():
                    q_value, hidden = self.model(agent_state)
                
                # epsilon-greedy策略选择动作
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = torch.argmax(q_value, 1).item()

                # apply action in env
                next_obs, reward, done, _ = self.env.step(action)

                # 将最新的观察值，动作，奖励，隐藏状态更新到AgentState中（采用替换的形式）
                # 而agent_state主要是给模型使用的
                agent_state.update(next_obs, action, reward, hidden)

                episode_steps += 1
                actor_steps += 1

                # 将动作记录到局部缓冲区
                self.local_buffer.add(action, reward, next_obs, q_value.numpy(), torch.cat(hidden).numpy())

                if done:
                    # 如果游戏结束，则打包一个生命周期内的所有数据
                    block = self.local_buffer.finish()
                    self.sample_queue.put(block)

                elif len(self.local_buffer) == self.block_length or episode_steps == self.max_episode_steps:
                    # 如果局部缓冲区的长度达到了block_length或者生命周期内的步数达到了最大步数
                    # 则中断采集，直接打包当前局部缓冲区的数据
                    with torch.no_grad():
                        q_value, hidden = self.model(agent_state)

                    block = self.local_buffer.finish(q_value.numpy())

                    if self.epsilon > 0.01:
                        '''
                        block[2]存储的是episode_reward
                        在探索阶段(epsilon较大时)，将episode_reward设为None
                        这是因为探索阶段的回报(reward)不能很好地反映策略的真实性能
                        只有当epsilon很小时(< 0.01)，agent主要依据学到的策略行动，此时的回报才更有参考价值
                        '''
                        block[2] = None
                    self.sample_queue.put(block)

                if actor_steps % 400 == 0:
                    # 每400步更新一次共享模型的权重
                    self.update_weights()

                
    def update_weights(self):
        '''load the latest weights from shared model'''
        self.model.load_state_dict(self.shared_model.state_dict())
    
    def reset(self):
        # 重置环境
        obs = self.env.reset()
        # 重置局部缓冲区，一个生命 周期的动作缓冲区
        self.local_buffer.reset(obs)

        # 待力气状态，包含了观察值，动作维度
        state = AgentState(torch.from_numpy(obs).unsqueeze(0), self.action_dim)

        return state

