# train the network based on replay_buffer
import tensorflow as tf
from deepmind import Network, ReplayBuffer, AlphaZeroConfig, NeuralNet
import time

def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  loss = 0
  for image, (target_value, target_policy) in batch:
    policy_logits, value = network.inference(image)
    loss += (
        tf.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=target_policy))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)

def train_network(replay_buffer: ReplayBuffer):
    nn = NeuralNet()
    nn.latest_network()
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                           config.momentum)

    for i in range(config.training_steps):
        replay_buffer.reload()
        batch = replay_buffer.sample_batch()
        while len(batch) == 0:
            print("wait 60 seconds...")
            time.sleep(60)
            replay_buffer.reload()
            batch = replay_buffer.sample_batch()

        # update_weights(optimizer, nn, batch, config.weight_decay)
        nn.train(batch)
        if i % config.checkpoint_interval == 0:
            nn.save_network()


# n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)
config: AlphaZeroConfig = AlphaZeroConfig()
replay_buffer = ReplayBuffer()
train_network(replay_buffer)
