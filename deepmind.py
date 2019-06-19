"""Pseudocode description of the AlphaZero algorithm."""

# from __future__ import division

import math
import os
import time
from threading import Lock
from typing import List

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import BatchNormalization, Reshape, Activation, Conv2D, Flatten, Dropout, Dense, MaxPooling2D
import pickle
import mcts2

# returns all moves possible from a board configuration
ValidMovesFromState = {}

# profiling
generate_state_from_history_counter = 0
getValidMoves_counter = 0
apply_counter = 0
do_action_counter = 0
make_image_counter = 0
undo_action_counter = 0


##########################
####### Helpers ##########


class AlphaZeroConfig(object):

    def __init__(self):
        ### Self-Play
        self.num_actors = 1

        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_simulations = 50

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
        self.epochs = 3

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }

        self.num_channels = 512
        self.dropout = 0.3
        self.lr = 0.001

        self.checkpoint = './temp/'
        self.filename = 'Deepmind2'
        self.replaybuffer = 'replay'


config: AlphaZeroConfig = AlphaZeroConfig()


class Node(object):

    def __init__(self, prior: float, description):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.description = description

    def __repr__(self):
        return f"s:{self.value():0.5f} vis:{self.visit_count} p:{self.prior:0.2f} c:{len(self.children)} to_play:{self.to_play} {self.description}"

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def setPrior(self, prior: float):
        self.prior = prior


class GameProtocol(object):

    def __init__(self, history=None):
        self.history = history or []
        self.child_visits = []

    def terminal(self):
        # Game specific termination rules.
        pass

    def terminal_value(self, to_play):
        # Game specific value - current score for the position,
        # should be called when game is already finished -1, 1, 0
        pass

    def legal_actions(self):
        # Game specific calculation of legal actions.
        return []

    def clone(self):
        return GameProtocol(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.NUM_ACTIONS)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes from specified move.
        return []

    def make_target(self, state_index: int):
        # tuple of game result and actions counts
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    @property
    def to_play(self):
        return len(self.history) % 2


# the input data for the neural network from the move number of the current game
# key-value:
# key - history
# image list generated for a game with that history
TrainImages = {}


class Moara(GameProtocol):
    VALID_MOVES = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.VALID_MOVES]
    MILLS = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.MILLS]
    BOARD_X = 8
    BOARD_Y = 3
    BOARD_SIZE = BOARD_X * BOARD_Y
    NUM_ACTIONS = (BOARD_SIZE +  # put at
                   BOARD_SIZE +  # put at which creates a mill (need capture) - there must be two of this in the history
                   BOARD_SIZE +  # moving from - there must be two of these
                   BOARD_SIZE  # capture from
                   )

    NUM_FEATURES = (1 +  # board for player 1
                    1 +  # board for player 2
                    1 +  # rotated board for player 1
                    1 +  # rotated board for player 2
                    1 +  # unused pieces for player 1
                    1 +  # unused pieces for player 2
                    1)  # color
    NUM_STEPS = 4  # keep current state and previous 7
    planes = (NUM_STEPS * NUM_FEATURES +
              # 1 +  # no of repetitions for current board, uniform value over array
              1)  # total moves, uniform value over array
    # indexes into internal state for various information
    UNUSED_PIECES_INDEX = BOARD_SIZE
    PLAYER_PIECES_INDEX = UNUSED_PIECES_INDEX + 2
    NO_MOVES_WITHOUT_CAPTURE_INDEX = PLAYER_PIECES_INDEX + 2
    NO_MOVES_INDEX = NO_MOVES_WITHOUT_CAPTURE_INDEX + 1
    LAST_LAST_ACTION_INDEX = NO_MOVES_INDEX + 1
    LAST_ACTION_INDEX = LAST_LAST_ACTION_INDEX + 1

    def __init__(self, history=None):
        super().__init__(history)
        if self.history == []:
            state = np.array([0] * (self.LAST_ACTION_INDEX + 1))
            state[self.UNUSED_PIECES_INDEX] = 9
            state[self.UNUSED_PIECES_INDEX + 1] = 9
            state[self.PLAYER_PIECES_INDEX] = 9
            state[self.PLAYER_PIECES_INDEX + 1] = 9
            state[self.LAST_LAST_ACTION_INDEX] = -1
            state[self.LAST_ACTION_INDEX] = -1
            self.history.append(state)
        # dynamic state
        # self.internalState = np.array([0] * self.BOARD_SIZE)
        # self.unusedPieces = [0, 0]
        # self.playerPieces = [0, 0]
        # self.noMovesWithoutCapture = 0
        # self.noMoves = 0

        # using current history, generate various states
        # for index, action in enumerate(self.history):
        #     self.do_action(action, index)
        # self.generate_state_from_history()

    def stateToString(state):
        _board = ''.join(['x' if x == 1 else 'o' if x == -1 else '_' for x in state[:Moara.BOARD_SIZE]])
        _player = 'x' if state[Moara.NO_MOVES_INDEX] % 2 == 0 else 'o'
        # moves = functools.reduce(lambda acc, i: f"{acc}.{i}", self.history, "")
        _last_action = state[Moara.LAST_ACTION_INDEX]
        _last_last_action = state[Moara.LAST_LAST_ACTION_INDEX]
        return f"{_board}{state[Moara.UNUSED_PIECES_INDEX]}{state[Moara.UNUSED_PIECES_INDEX + 1]}{_player} [{_last_last_action}:{_last_action}]"

    # string representation of the current position in the game
    def __repr__(self):
        return Moara.stateToString(self.history[-1])

    def clone(self):
        return Moara(list(self.history))

    @property
    def to_play(self):
        return self.history[-1][self.NO_MOVES_INDEX] % 2

    @property
    def internalState(self):
        return self.history[-1]

    @property
    def unusedPieces(self):
        return self.internalState[self.UNUSED_PIECES_INDEX:self.UNUSED_PIECES_INDEX + 2]

    @property
    def playerPieces(self):
        return self.internalState[self.PLAYER_PIECES_INDEX:self.PLAYER_PIECES_INDEX + 2]

    @property
    def noMovesWithoutCapture(self):
        return self.internalState[self.NO_MOVES_WITHOUT_CAPTURE_INDEX]

    @noMovesWithoutCapture.setter
    def noMovesWithoutCapture(self, value):
        self.internalState[self.NO_MOVES_WITHOUT_CAPTURE_INDEX] = value

    @property
    def noMoves(self):
        return self.internalState[self.NO_MOVES_INDEX]

    @noMoves.setter
    def noMoves(self, value):
        self.internalState[self.NO_MOVES_INDEX] = value

    @property
    def last_action(self):
        return self.internalState[self.LAST_ACTION_INDEX]

    @last_action.setter
    def last_action(self, value):
        self.internalState[self.LAST_LAST_ACTION_INDEX] = self.history[-1][self.LAST_ACTION_INDEX]
        self.internalState[self.LAST_ACTION_INDEX] = value

    @property
    def last_last_action(self):
        return self.internalState[self.LAST_LAST_ACTION_INDEX]

    def decUnusedPlayerCount(self, player):
        self.unusedPieces[(player + 1) // 2] -= 1

    def generate_state_from_history(self):
        return
        start = time.clock()
        self.noMoves = len(self.history)
        # calculate the number of unused pieces:
        # for each move from the beginning a piece must be put,
        # so after 18 moves, all must be used
        # but removing any capture moves (this come in pairs)
        only_put_actions: int = len(set([action for action in self.history if action < 2 * self.BOARD_SIZE]))
        self.unusedPieces = [max(0, 9 - (only_put_actions + 1) // 2), max(0, 9 - (only_put_actions + 0) // 2)]

        self.playerPieces = [
            len([0 for pos in range(self.BOARD_SIZE) if self.internalState[pos] == 1]) + self.unusedPieces[0],
            len([0 for pos in range(self.BOARD_SIZE) if self.internalState[pos] == -1]) + self.unusedPieces[1]]

        # count moves until a capture move is made
        list_of_captures = [self.history.index(x) for x in self.history if x > self.BOARD_SIZE * 3]
        self.noMovesWithoutCapture = (len(self.history) - 1 - list_of_captures[-1]) if len(
            list_of_captures) > 0 else len(self.history)
        end = time.clock()
        global generate_state_from_history_counter
        generate_state_from_history_counter += (end - start)

    def getValidMoves(self):
        global getValidMoves_counter
        start = time.clock()
        # memoization
        s = str(self)
        if s in ValidMovesFromState:
            # if s not in invariantBoard.history or invariantBoard.history[s] < 2:
            result = ValidMovesFromState[s]
            end = time.clock()
            getValidMoves_counter += (end - start)
            return result
            # else:
        result_1 = []
        result = []
        need_to_pass = self.BOARD_SIZE <= self.last_action < self.BOARD_SIZE * 3 and self.last_last_action != self.last_action
        if need_to_pass:
            end = time.clock()
            getValidMoves_counter += (end - start)
            return [self.last_action]
        need_to_select_move = self.BOARD_SIZE * 2 <= self.last_action < self.BOARD_SIZE * 3 and self.last_action == self.last_last_action
        need_to_finish_capture = self.BOARD_SIZE * 1 <= self.last_action < self.BOARD_SIZE * 2 and self.last_action == self.last_last_action
        need_to_put = not need_to_finish_capture and self.unusedPieces[self.to_play] > 0
        player_color = 1 if self.to_play == 0 else -1

        if need_to_put:  # put
            # phase 1: can put anywhere where there is an empty place
            result_1 = [x for x in range(self.BOARD_SIZE) if self.internalState[x] == 0]
        elif need_to_finish_capture:
            # select an opponent not in a mill
            all_opponent_pieces = [x for x in range(self.BOARD_SIZE) if self.internalState[x] == -player_color]
            # that are not in an enemy mill
            available_opponent_pieces = list(filter(lambda p: self.isInAMill(p, -player_color) is False,
                                                    all_opponent_pieces))
            if len(available_opponent_pieces) == 0:
                # if no available enemy piece to capture outside of aa mill
                # retry with a piece from an enemy mill
                # invariantBoard.display()
                available_opponent_pieces = all_opponent_pieces
            # for each available opponent piece to be captured
            result = [x + self.BOARD_SIZE * 3 for x in available_opponent_pieces]
        elif need_to_select_move:
            # select moves that start from the position and are empty
            if self.playerPieces[self.to_play] > 3:
                valid_moves = list(filter(lambda x: x[0] == self.last_action % self.BOARD_SIZE, self.VALID_MOVES))
                # transform into index in action moves
                result_1 = [v_m for v_m in set([x[1] for x in valid_moves]) if self.internalState[v_m] == 0]
            else:
                # jump - all free places
                result_1 = [x for x in range(self.BOARD_SIZE) if self.internalState[x] == 0]
        else:
            # move - select all positions from which a move can be started
            if self.playerPieces[self.to_play] > 3:  # move
                valid_moves = list(
                    filter(lambda x: self.internalState[x[0]] == player_color and self.internalState[x[1]] == 0,
                           self.VALID_MOVES))
                result = [v_m + 2 * self.BOARD_SIZE for v_m in set([x[0] for x in valid_moves])]
            else:  # jump
                remaining = [x for x in range(self.BOARD_SIZE) if self.internalState[x] == player_color]
                result = [r + 2 * self.BOARD_SIZE for r in remaining]
        # for each possible move, determine if it forms a mill
        # if so then can capture any of the opponent pieces, that are not in a mill
        for possible_action in result_1:
            # there is also an origin to the move:
            # if (move % self.possibleMovesSize) > self.boardSize:
            #     orig = (move % self.possibleMovesSize) // self.boardSize - 1;
            mills = self.getMills(possible_action)  # any pos is in two mills
            wouldBeInMill = False
            for mill in mills:
                # copy_of_internal_array = np.array(self.internalState)
                # copy_of_internal_array[possible_action] = player_color
                # if orig != -1:
                #     copy_of_internal_array[orig] = 0
                # sum_mill = copy_of_internal_array[mill[0]] + \
                #            copy_of_internal_array[mill[1]] + \
                #            copy_of_internal_array[mill[2]]
                sum_mill = self.internalState[mill[0]] + self.internalState[mill[1]] + self.internalState[mill[2]]
                if player_color +  sum_mill == 3 * player_color:
                    wouldBeInMill = True
                    break

            if wouldBeInMill:
                # need to signal that this move will require a capture
                result.append(possible_action + 1 * self.BOARD_SIZE)
            else:  # no mill, no capture
                result.append(possible_action)

        ValidMovesFromState[s] = result
        # self.SaveValidMoves()
        end = time.clock()
        getValidMoves_counter += (end - start)
        return result

    def apply(self, action):
        global apply_counter
        start = time.clock()
        category = action // self.BOARD_SIZE
        move = action % self.BOARD_SIZE
        self.history.append(np.array(self.history[-1]))
        player_color = 1 if self.to_play == 0 else -1

        if category == 0:
            if self.last_action != -1 and self.last_action == self.last_last_action:  # part 2 of a move
                origin = self.last_action % self.BOARD_SIZE
                if self.internalState[origin] != player_color:
                    NOP = 0
                assert (self.internalState[origin] == player_color)
                if self.internalState[move] != 0:
                    NOP = 0
                assert (self.internalState[move] == 0)
                self.internalState[origin] = 0
                self.internalState[move] = player_color
            else:  # put a piece from unused
                assert (self.internalState[move] == 0)
                self.internalState[move] = player_color
                self.unusedPieces[self.to_play] -= 1
                self.noMovesWithoutCapture += 1
        elif category == 1:
            if self.last_action == action:
                # the move is for the opponent, pass
                self.noMovesWithoutCapture += 1
            elif self.last_action == self.last_last_action:  # move & prepare capture
                origin = self.last_action % self.BOARD_SIZE
                assert (self.internalState[origin] == player_color)
                assert (self.internalState[move] == 0)
                self.internalState[origin] = 0
                self.internalState[move] = player_color
            else:
                # put the piece, and an opponent pieces will be selected for removal
                assert (self.internalState[move] == 0)
                self.internalState[move] = player_color
                self.unusedPieces[self.to_play] -= 1
                self.noMovesWithoutCapture += 1
        elif category == 3:
            # eliminate a piece
            assert (self.internalState[move] == -player_color)
            self.internalState[move] = 0
            self.noMovesWithoutCapture = 0  # reset it
            self.playerPieces[0 if self.to_play == 1 else 1] -= 1
        else:  # prepare move to destination
            if self.last_action == action:
                # the move is for the opponent, pass
                self.noMovesWithoutCapture += 1
            else:  # move for player, step 1, do nothing
                self.noMovesWithoutCapture += 1
                assert (self.internalState[action % self.BOARD_SIZE] == player_color)
        # self.history.append(action)
        self.noMoves += 1
        self.last_action = action
        end = time.clock()
        apply_counter += (end - start)
        return

    def isInAMill(self, pos, player):
        # find all mills that contain the pos
        mill_list = self.getMills(pos)
        return list(filter(lambda x: self.getMillSum(x) == 3 * player, mill_list)) != []

    # get the mills that contain position
    def getMills(self, pos):
        return list(filter(lambda mill: list(filter(lambda p: p == pos, mill)) != [], self.MILLS))

    # get the sum of content of the mill
    def getMillSum(self, mill):
        return (self.internalState[mill[0]] +
                self.internalState[mill[1]] +
                self.internalState[mill[2]])

    def legal_actions(self):
        # Game specific calculation of legal actions.
        return self.getValidMoves()

    def do_action(self, action, state_index):
        global do_action_counter
        start = time.clock()
        category = action // self.BOARD_SIZE
        move = action % self.BOARD_SIZE
        last_action = self.history[state_index - 1] if state_index > 0 else -1
        last_last_action = self.history[state_index - 2] if state_index > 1 else -1
        # put piece
        if category == 0 or category == 1:
            if action != last_action:
                self.internalState[move] = 1 if state_index % 2 == 0 else -1
                # if it's part of a move
                if last_action // self.BOARD_SIZE == 2 and last_action == last_last_action:
                    self.internalState[last_action % self.BOARD_SIZE] = 0
        # remove piece from location
        if category == 3:
            if action != last_action:
                self.internalState[move] = 0
        end = time.clock()
        do_action_counter += (end - start)

    def undo_action(self, action):
        global undo_action_counter
        start = time.clock()
        category = action // self.BOARD_SIZE
        move = action % self.BOARD_SIZE
        last_action = self.history[-2] if self.noMoves > 1 else -1
        last_category = last_action // self.BOARD_SIZE
        player = 1 if len(self.history) % 2 == 0 else -1
        # put piece -> remove piece
        if category == 0 or category == 1 and action != last_action:
            self.internalState[move] = 0
            # if it's part of a move
            if last_action // self.BOARD_SIZE == 2:
                self.internalState[move] = player
        # put piece back
        if last_category == 2 and action != last_action:  # put own piece back
            self.internalState[move] = player
        if category == 3 and action != last_action:  # put opponent piece back
            self.internalState[move] = -player

        self.history = self.history[:-1]
        self.generate_state_from_history()
        end = time.clock()
        undo_action_counter += (end - start)

    def make_image(self, state_index: int):
        global make_image_counter
        start = time.clock()
        result = []
        if state_index != -1:
            s = Moara.stateToString(self.history[state_index])
            if s in TrainImages:
                result = TrainImages[s]
                end = time.clock()
                make_image_counter += (end - start)
                return result
        else:
            if str(self) in TrainImages:
                result = TrainImages[str(self)]
                end = time.clock()
                make_image_counter += (end - start)
                return result

        # Game specific feature planes.
        # if state is -1 return use current position
        # otherwise generate from the specified position
        if state_index != -1:
            # generate a clone up to the specified state_index
            local = self.history[: min(state_index, len(self.history))]
        else:
            local = self.history
        noMovesWithoutCapture = local[-1][self.NO_MOVES_WITHOUT_CAPTURE_INDEX]
        # propagate back in temporal array
        for i in range(self.NUM_STEPS):  # i = 0, is the oldest
            # use historical moves, extract from history, if enough moves where made
            if local[-1][self.NO_MOVES_INDEX] < i:
                result += [[0] * self.BOARD_SIZE for _ in range(self.NUM_FEATURES)]
            else:
                if i > 0:
                    # action = local.history[-1]
                    # # undo action on the current board
                    # local.undo_action(action)
                    local = local[:-1]

                # normal board for player 1
                result.append([1 if x == 1 else 0 for x in local[-1][:self.BOARD_SIZE]])
                # normal board for player 2
                result.append([1 if x == -1 else 0 for x in local[-1][:self.BOARD_SIZE]])
                arr2 = np.array(self.MILLS).reshape(48)
                # rotate board
                rot = [local[-1][arr2[x + self.BOARD_SIZE]] for x in range(self.BOARD_SIZE)]
                # rotated board for player 1
                result.append([1 if x == 1 else 0 for x in rot])
                # rotated board for player 2
                result.append([1 if x == -1 else 0 for x in rot])

                # local.generate_state_from_history()
                # unused pieces for player 1
                result.append([local[-1][self.UNUSED_PIECES_INDEX]] * self.BOARD_SIZE)
                # unused pieces for player 2
                result.append([local[-1][self.UNUSED_PIECES_INDEX + 1]] * self.BOARD_SIZE)

                # player at move
                result.append([local[-1][self.NO_MOVES_INDEX] % 2] * self.BOARD_SIZE)

        # # repetition
        # s = str(self)
        # rep = self.history[s] if s in self.history else 0
        # result.append([rep] * self.boardSize)
        # # total moves
        # result.append([self.noMoves] * self.boardSize)
        # total moves without capture
        result.append([noMovesWithoutCapture] * self.BOARD_SIZE)

        TrainImages[str(self)] = result
        end = time.clock()
        make_image_counter += (end - start)
        return result

    def terminal(self):
        # Game specific termination rules.
        if self.noMovesWithoutCapture > 100:
            return True
        if self.playerPieces[0] < 3:
            return True
        if self.playerPieces[1] < 3:
            return True
        if not self.getValidMoves():
            # self.display()
            return True
        return False

    def terminal_value(self, to_play):
        # Game specific value - current score for the position,
        # should be called when game is already finished -1, 1, 0
        if self.playerPieces[0] < 3:
            return -1 if to_play == 0 else 1
        if self.playerPieces[1] < 3 or self.getValidMoves():
            return 1 if to_play == 0 else -1
        # draw
        return 0


class ReplayBuffer(object):

    def __init__(self):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

        folder = config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, config.replaybuffer + ".data")
        with open(filename, "wb+") as f:
            pickle.Pickler(f).dump(self.buffer)
        f.closed

    def reload(self):
        modelFile = os.path.join(config.checkpoint, config.replaybuffer + ".data")
        if not os.path.isfile(modelFile):
            print(modelFile)
            print("File with trainExamples not found.")
        else:
            print("File with trainExamples found. Read it.")
            with open(modelFile, "rb") as f:
                self.buffer = pickle.Unpickler(f).load()
            print(f"Found {len(self.buffer)} games")
            f.closed


    def sample_batch(self):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history)-1 for g in self.buffer))
        if move_sum == 0.0:
            return []
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[(len(g.history) - 1)/ move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history)-1)) for g in games]
        return [(g.make_image(i+1), g.make_target(i)) for (g, i) in game_pos]


class Network(object):

    def inference(self, image):
        return (-1, {})  # Value, Policy

    def get_weights(self):
        # Returns the weights of this network.
        return []


class NeuralNet(Network):
    def __init__(self):
        self.lock = Lock()
        self.action_size = Moara.NUM_ACTIONS
        self.board_x = Moara.BOARD_X  # lines
        self.board_y = Moara.BOARD_Y  # columns
        self.planes = Moara.planes

        self.input_boards = Input(shape=(self.planes, self.board_x, self.board_y))  # s: batch_size x board_x x board_y

        self.x_image = BatchNormalization(axis=3)(Reshape((self.board_x, self.board_y, self.planes))(
            self.input_boards))  # batch_size  x board_x x board_y x 1
        self.h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(
            self.x_image)))  # batch_size  x board_x x board_y x num_channels
        self.pool = MaxPooling2D()(self.h_conv1)
        self.h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(
            self.pool)))  # batch_size  x board_x x board_y x num_channels
        # self.h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same')(
        #     self.h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
        # self.h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='valid')(
        #     self.h_conv3)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        self.h_conv4_flat = Flatten()(self.h_conv2)
        self.s_fc1 = Dropout(config.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(self.h_conv4_flat))))  # batch_size x 1024
        self.s_fc2 = Dropout(config.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512)(self.s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(self.s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(self.s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.optimizer = tf.train.MomentumOptimizer(0.2,
                                                    config.momentum)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], loss_weights=[1., 5.],
                           optimizer=self.optimizer)
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
        #
        inputData = inputData[np.newaxis, :, :]
        with self.lock:
            pi, v = self.model.predict(inputData)
        return pi[0], v[0][0]

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
        input_boards, targets = list(zip(*examples))
        target_vs, target_pis = list(zip(*targets))
        input_boards = np.array(input_boards).reshape(len(input_boards), self.planes, self.board_x, self.board_y)

        print("mumu")
        print(len(examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        result = self.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=1024,
                                epochs=config.epochs, validation_split=0.1)
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
        with self.lock:
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
        with self.lock:
            filename = f"no{filename_no}.neural.data"
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                self.model.load_weights(filepath)
            else:
                print("No model in path '{}'".format(filepath))

    def get_weights(self):
        # Returns the weights of this network.
        return self.model.weights

    def latest_network(self):
        # return new network
        self.load_checkpoint(folder=config.checkpoint, filename_no=config.filename)

    def save_network(self, step: int = -1):
        if step != -1:
            self.save_checkpoint(folder=config.checkpoint, filename_no=f"{config.filename}.{step}")
        else:
            self.save_checkpoint(folder=config.checkpoint, filename_no=config.filename)


##### End Helpers ########
##########################


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(replay_buffer: ReplayBuffer):
    nn = NeuralNet()
    while True:
        nn.latest_network()
        game = play_game(nn)
        replay_buffer.save_game(game)
        # train_network(replay_buffer)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(network: Network):
    game = Moara()
    while not game.terminal() and game.noMoves < config.max_moves:
        action, root = run_mcts(game, network)
        game.apply(action)
        print(f"{len(game.history):03d} {game.playerPieces[0]}:{game.playerPieces[1]} {root.value():0.2f} {game}")
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(game: GameProtocol, network: Network):
    root = Node(0, "")
    evaluate(root, game, network)
    add_exploration_noise(root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(node)
            scratch_game.apply(action)
            search_path.append(node)

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play)
    return select_action(game, root), root


def select_action(game: GameProtocol, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(node: Node):
    scores = [(ucb_score(node, child), action, child) for action, child in node.children.items()]
    _, action, child = max(scores)
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node):
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
    node.to_play = game.to_play
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    node.children = {action: Node(p / policy_sum, f"{node.description}.{action}") for action, p in policy.items()}
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.visit_count += 1
        # from -1 to +1
        # node.value_sum += value if node.to_play == to_play else -value
        # from 0 to 1
        node.value_sum += value if node.to_play == to_play else (1 - value)


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    # for a, n in zip(actions, noise):
    #     node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    [node.children[a].setPrior(node.children[a].prior * (1 - frac) + n * frac) for a, n in zip(actions, noise)]


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(replay_buffer: ReplayBuffer):
    # old_train_network(replay_buffer)
    globalNeuralNet.latest_network()
    for i in range(config.training_steps):
        print(f"training step {i} out of {config.training_steps}")
        if i % config.checkpoint_interval == 0:
            globalNeuralNet.save_network(i)
        batch = replay_buffer.sample_batch()
        while batch == []:
            time.sleep(60)
        if batch != []:
            globalNeuralNet.train(batch)
            globalNeuralNet.save_network()

def old_train_network(replay_buffer: ReplayBuffer):
  # network = Network()
  optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                         config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
        globalNeuralNet.save_network(i)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, globalNeuralNet, batch, config.weight_decay)
    globalNeuralNet.save_network()


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
    loss = 0
    loop = 0
    for image, (target_value, target_policy) in batch:
        print(f"loop {loop}")
        loop += 1
        policy_logits, value  = network.inference(image)
        loss_v = tf.losses.mean_squared_error(value, target_value)
        loss_p = tf.nn.softmax_cross_entropy_with_logits(
                    logits=policy_logits, labels=target_policy)
        loss += (loss_v + loss_p)

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
    # normalization
    probabilities = [x[0] / config.num_simulations for x in d]
    res = np.random.choice(len(probabilities), p=probabilities)
    return d[res]


def launch_job(f, *args):
    f(*args)
    # thread = Thread(target=f, args=args)
    # thread.start()
    # thread.join()


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero():
    replay_buffer = ReplayBuffer()

    for i in range(config.num_actors):
        launch_job(run_selfplay, replay_buffer)

    train_network(replay_buffer)

if __name__ == "__main__":
    # moaraGame = Moara()
    # n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)
    config: AlphaZeroConfig = AlphaZeroConfig()
    alphazero()
