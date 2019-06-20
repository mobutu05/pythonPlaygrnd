
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import deepmind

config: deepmind.AlphaZeroConfig = deepmind.AlphaZeroConfig()
replay_buffer = deepmind.ReplayBuffer()
replay_buffer.reload()

# for i in range(config.num_actors):
deepmind.run_selfplay(replay_buffer)
