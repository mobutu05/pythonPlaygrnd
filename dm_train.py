# train the network based on replay_buffer
import tensorflow as tf
import deepmind


def train_network(replay_buffer: deepmind.ReplayBuffer):
    nn = deepmind.NeuralNet()
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                           config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            nn.save_network(i)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


# n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)
config: deepmind.AlphaZeroConfig = deepmind.AlphaZeroConfig()
train_network()
