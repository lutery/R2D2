import random
import torch.multiprocessing as mp
import torch
import numpy as np
from worker import Learner, Actor, ReplayBuffer
from environment import create_env
from model import Network
import config

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)

def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha, num_actors: int = config.num_actors):
    '''
    actor_id: 动作器的ID，因为有多个动作器，不同动作器之间的id不同
    base_eps: 基础的epsilon值
    alpha: 用于计算epsilon的参数
    num_actors: 动作器的数量
    '''
    
    # 得到不同的actor_id对应的epsilon值
    # 1 + （ 1 / ( 8 - 1) ) * 0.4 = 1.0571428571428572
    exponent = 1 + actor_id / (num_actors-1) * alpha
    return base_eps**exponent


def train(num_actors=config.num_actors, log_interval=config.log_interval):

    model = Network(create_env().action_space.n)
    model.share_memory() # 设置共享模型，可以自动在不同的进程之间共享模型参数
    # 看起来要搞多进程训练
    sample_queue_list = [mp.Queue() for _ in range(num_actors)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(sample_queue_list, batch_queue, priority_queue)
    learner = Learner(batch_queue, priority_queue, model)
    # 创建动作器的列表
    actors = [Actor(get_epsilon(i), model, sample_queue_list[i]) for i in range(num_actors)]

    # 将每个动作器分配给一个进程，执行的函数就是actor.run，并且在run里面可以访问actor的成员变量
    actor_procs = [mp.Process(target=actor.run) for actor in actors]
    for proc in actor_procs:
        proc.start()

    # 这个run函数是什么的？
    buffer_proc = mp.Process(target=buffer.run)
    buffer_proc.start()

    # 开始训练
    learner.run()

    # 等待所有进程结束，也就是等待训练结束
    buffer_proc.join()

    # 关闭所有的动作器进程
    for proc in actor_procs:
        proc.terminate()


if __name__ == '__main__':

    train()

