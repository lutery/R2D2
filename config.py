game_name = 'MsPacman' # 游戏名称，比如breakout
obs_shape = (1, 84, 84)

lr = 1e-4
eps = 1e-3
grad_norm = 40
batch_size = 64
learning_starts = 50000
save_interval = 500
target_net_update_interval = 2000
gamma = 0.997
prio_exponent = 0.9
importance_sampling_exponent = 0.6

training_steps = 100000
buffer_capacity = 2000000 # 缓冲区的大小
max_episode_steps = 27000 # 每个环境的最大步数
actor_update_interval = 400
block_length = 400  # cut one episode to numbers of blocks to improve the buffer space utilization

num_actors = 8 # 有多少个动作器用于数据采集
base_eps = 0.4 # 起始的epsilon值
alpha = 7
log_interval = 10 # 这个应该时记录的日志周期

# sequence setting
burn_in_steps = 40
learning_steps = 40
forward_steps = 5
seq_len = burn_in_steps + learning_steps + forward_steps

# network setting
hidden_dim = 512

render = False
save_plot = True
test_epsilon = 0.001
