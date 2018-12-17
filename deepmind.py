"""Pseudocode description of the AlphaZero algorithm."""

from __future__ import division

import math
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import BatchNormalization, Reshape, Activation, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from typing import List
import mcts2
import moara
import os

ValidMovesFromState = {}


##########################
####### Helpers ##########


class AlphaZeroConfig(object):

    def __init__(self):
        ### Self-Play
        self.num_actors = 5000

        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_simulations = 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class GameProtocol(object):

    def __init__(self, history=None):
        self.history = history or []
        self.child_visits = []
        self.num_actions = 4672  # action space size for chess; 11259 for shogi, 362 for Go

    def terminal(self):
        # Game specific termination rules.
        pass

    def terminal_value(self, to_play):
        # Game specific value.
        pass

    def legal_actions(self):
        # Game specific calculation of legal actions.
        return []

    def clone(self):
        return GameProtocol(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        return []

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2


class Moara(GameProtocol):
    VALID_MOVES = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.VALID_MOVES]
    MILLS = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.MILLS]

    def __init__(self, history=None):
        super().__init__(history)
        self.board_X = 8
        self.board_Y = 3
        self.boardSize = self.board_X * self.board_Y
        # 1 if capture is to be made by current player
        # -1 if capture is to be made by opponent
        # 0 no capture
        self.boardStatus = 0
        self.unusedPieces = [9, 9]  # for opponent, player
        self.playerAtMove = 1
        self.internalArray = np.array([0] * self.boardSize)
        # list of moves
        self.moves = []
        # number of moves since last capture
        self.noMovesWithoutCapture = 0
        # number of total moves
        self.noMoves = 0
        self.time_len = 4  # keep current state and previous 7
        self.feature_len = 0  # number of planes describing current board
        self.num_actions = (self.boardSize +  # piece position
                            # self.boardSize * self.boardSize +  # move/jump pieces
                            1)  # pass
        self.moves.append(str(self))

    # string representation of the current position in the game
    def __repr__(self):
        board = ''.join(['x' if x == 1 else 'o' if x == -1 else '_' for x in self.internalArray])
        player = 'x' if self.playerAtMove == 1 else 'o'
        statusSign = 'x' if self.boardStatus >= 1 else 'o' if self.boardStatus <= -1 else '_'
        statusValue = abs(self.boardStatus)
        return f"{board}{self.getUnusedPlayerCount(1)}{self.getUnusedPlayerCount(-1)}{player}{statusSign}{statusValue:02d}"

    def getUnusedPlayerCount(self, player):
        return self.unusedPieces[(player + 1) // 2]

    def getPlayerCount(self, player):
        no_pieces_on_board = len([0 for pos in range(self.boardSize) if self.internalArray[pos] == player])
        return no_pieces_on_board + self.getUnusedPlayerCount(player)

    def getValidMoves(self, player):
        orig = -1
        dest = -1

        # memoization
        s = str(self)
        if s in ValidMovesFromState:
            # if s not in invariantBoard.history or invariantBoard.history[s] < 2:
            return ValidMovesFromState[s]
            # else:
        result = []
        moves = []
        capture = []
        if self.boardStatus != 0 and np.sign(self.boardStatus) == -player:
            result = [self.boardSize]  # pass
            ValidMovesFromState[s] = result
            # self.SaveValidMoves()
            return result
        boardStatus = abs(self.boardStatus)
        if boardStatus == 0:
            if self.getUnusedPlayerCount(player) > 0:  # put
                # phase 1: can put anywhere where there is an empty place
                result = [x for x in range(self.boardSize) if self.internalArray[x] == 0]
            else:
                if self.getPlayerCount(player) > 3:  # move
                    # select starting points for the move
                    valid_moves = list(filter(lambda x: self.internalArray[x[0]] == player and
                                                        self.internalArray[x[1]] == 0,
                                              self.VALID_MOVES))
                    result = [x[0] for x in valid_moves]
                else:  # jump
                    result = [x for x in range(self.boardSize) if self.internalArray[x] == player]
        else:
            if boardStatus == self.boardSize + 1:
                # find all opponents available to capture
                all_opponent_pieces = [x for x in range(self.boardSize) if self.internalArray[x] == -player]
                # that is not an enemy mill
                available_opponent_pieces = list(
                    filter(lambda p: self.isInAMill(p, -player) is False, all_opponent_pieces))
                if len(available_opponent_pieces) == 0:
                    # if no available enemy piece to capture outside of aa mill
                    # retry with a piece from an enemy mill
                    # invariantBoard.display()
                    available_opponent_pieces = all_opponent_pieces
                # for each available opponent piece to be captured
                result = available_opponent_pieces
            else:
                # move or capture, and the status contains the destination
                if self.getPlayerCount(player) > 3:  # move
                    # self.display()
                    if self.internalArray[boardStatus - 1] != player:
                        assert (self.internalArray[boardStatus - 1] == player)
                    # select all moves that start from the (status - 1)
                    valid_moves = list(
                        filter(lambda x: x[0] == boardStatus - 1 and self.internalArray[x[1]] == 0, self.VALID_MOVES))
                    # destination point
                    result = [x[1] for x in valid_moves]
                else:  # jump
                    result = [x for x in range(self.boardSize) if self.internalArray[x] == 0]

        ValidMovesFromState[s] = result
        # self.SaveValidMoves()
        return result

    def legal_actions(self):
        # Game specific calculation of legal actions.
        return self.getValidMoves(1)

    def make_image(self, state_index: int):
        # Game specific feature planes.
        return self.getInternalRepresentation()

    def getInternalRepresentation(self):
        self.feature_len = (1 +  # board for player 1
                            1 +  # board for player 2
                            1 +  # rotated board for player 1
                            1 +  # rotated board for player 2
                            1 +  # unused pieces for player 1
                            1 +  # unused pieces for player 2
                            1 +  # color
                            1)  # board status
        result = []
        planes = (self.time_len * self.feature_len +
                  # 1 +  # no of repetitions for current board, uniform value over array

                  1)  # total moves, uniform value over array

        # propagate back in temporal array
        for i in range(self.time_len):  # i = 0, is the oldest
            # use historical moves, extract from history, if enough moves where made
            if len(self.moves) <= i:
                for j in range(self.feature_len):
                    result.append([0] * self.boardSize)
            else:
                l = list(self.moves[len(self.moves) - 1 - i])
                board = [1 if l[x] == 'x' else -1 if l[x] == 'o' else 0 for x in range(self.boardSize)]

                # normal board for player 1
                result.append([1 if x == 1 else 0 for x in board])
                # normal board for player 2
                result.append([1 if x == -1 else 0 for x in board])
                arr2 = np.array(self.MILLS).reshape(48)
                # rotate board
                rot = [board[arr2[x + self.boardSize]] for x in range(self.boardSize)]
                # rotated board for player 1
                result.append([1 if x == 1 else 0 for x in rot])
                # rotated board for player 2
                result.append([1 if x == -1 else 0 for x in rot])

                unusedPlayer1 = int(l[self.boardSize])
                unusedPlayer2 = int(l[self.boardSize + 1])
                # unused pieces for player 1
                result.append([unusedPlayer1] * self.boardSize)
                # unused pieces for player 2
                result.append([unusedPlayer2] * self.boardSize)

                player = 1 if l[self.boardSize + 2] == 'x' else -1
                statusSign = 1 if l[self.boardSize + 3] == 'x' else -1 if l[self.boardSize + 3] == 'o' else 0
                statusValue = int(l[self.boardSize + 4]) * 10 + int(l[self.boardSize + 5])
                status = statusValue * statusSign
                # player at move
                result.append([player] * self.boardSize)
                # board status
                result.append([status] * self.boardSize)

        # # repetition
        # s = str(self)
        # rep = self.history[s] if s in self.history else 0
        # result.append([rep] * self.boardSize)
        # # total moves
        # result.append([self.noMoves] * self.boardSize)
        # total moves without capture
        result.append([self.noMovesWithoutCapture] * self.boardSize)

        return result

    def terminal(self):
        # Game specific termination rules.
        if self.noMovesWithoutCapture > 100:
            return True  # 0.00000000001  # draw
        if self.getPlayerCount(self.playerAtMove) < 3:
            return True  # -1 * self.playerAtMove
        if self.getPlayerCount(-self.playerAtMove) < 3:
            return True  # 1 * self.playerAtMove
        player_valid_moves_list = self.getValidMoves(self.playerAtMove)
        if player_valid_moves_list == []:
            # self.display()
            return True  # -1 * self.playerAtMove
        return False


class ReplayBuffer(object):

    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class Network(object):

    def inference(self, image):
        return (-1, {})  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return []


class NeuralNet(Network):
    def __init__(self, game:GameProtocol, args):
        self.args = args
        self.action_size = game.num_actions
        self.board_x = game.board_X  # lines
        self.board_y = game.board_Y  # columns
        self.planes = len(game.getInternalRepresentation())

        self.input_boards = Input(shape=(self.planes, self.board_x, self.board_y))  # s: batch_size x board_x x board_y

        x_image = BatchNormalization(axis=3)(Reshape((self.board_x, self.board_y, self.planes))(
            self.input_boards))  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
            x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
            h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
            h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='valid')(
            h_conv3)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(self.args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(self.args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], loss_weights=[1., 5.],
                           optimizer=Adam(self.args.lr))
        print(self.model.summary())

    def inference(self, image):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        inputData = np.array(image).reshape(self.planes, self.board_x, self.board_y)

        inputData = inputData[np.newaxis, :, :]
        pi, v = self.model.predict(inputData)
        return pi[0], v[0]

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """

        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.array(input_boards).reshape(len(input_boards), self.planes, self.board_x, self.board_y)

        print("mumu")
        print(len(examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        result = self.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.args.batch_size,
                                epochs=self.args.epochs, validation_split=0.1)
        # for key in result.history.keys():
        #     print(key)
        #     print(result.history[key])

        # plt.clf()
        # epochs = range(1, len(result.history['loss']) + 1)
        # plt.plot(epochs, result.history['loss'], 'bo', label='Training acc')
        # plt.plot(epochs, result.history['val_loss'], 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # # plt.show()
        # plt.ion()
        # plt.show()
        # # plt.draw()
        # plt.pause(0.001)

    def save_checkpoint(self, folder, filename_no):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filename = f"no{filename_no}.neural.data"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder, filename_no):
        """
        Loads parameters of the neural network from folder/filename
        """
        'no37.neural.data'
        filename = f"no{filename_no}.neural.data"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
        else:
            print("No model in path '{}'".format(filepath))


class SharedStorage(object):

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.iterkeys())]
        else:
            return make_uniform_network()  # policy -> uniform, value -> 0.5

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


##### End Helpers ########
##########################



##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
    game = Moara()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: GameProtocol, network: Network):
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: GameProtocol, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.iteritems()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: GameProtocol, network: Network):
    policy_logits, value = network.inference(game.make_image(-1))

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network()
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                           config.momentum)
    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
    loss = 0
    for image, (target_value, target_policy) in batch:
        value, policy_logits = network.inference(image)
        loss += (
                tf.losses.mean_squared_error(value, target_value) +
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=policy_logits, labels=target_policy))

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
    return 0, 0


def launch_job(f, *args):
    f(*args)


def make_uniform_network():
    return n


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for i in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


moaraGame = Moara()
n = NeuralNet(moaraGame, mcts2.moara.args)
n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)

alphazero(AlphaZeroConfig())
