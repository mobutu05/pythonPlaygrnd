import functools
import math
import os
from collections import deque
from random import shuffle

import keras
import numpy as np
from keras import Input, Model
from keras.layers import BatchNormalization, Reshape, Activation, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import moara
import copy

trainExamplesHistory = []


# interface for game classes
class IGame:

    #if there is some state to be save to a persistent medium
    def SaveData(self):
        pass

    #any extra reward beside the one predicted by end game
    def getExtraReward(self):
        pass

    def reset(self):
        pass

    # display the board
    def display(self):
        pass

    def getCrtPlayer(self) -> int:
        pass

    def getPlayerCount(self, player)->int:
        pass

    def getGameEnded(self) -> float:
        """
        Input:
            canonical: whether to use the cannonical form, i.e. player invariant

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def getInternalRepresentation(self):
        """
        get internal representation of the game state, board to be used by nn
        :return: np array
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getValidMoves(self, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def getNextState(self, action):
        """
        Input:
            action: action taken by current player

        Returns:
            next: game after applying action
        """
        pass

    def getCanonicalForm(self):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, pi):
        """
        Input:
            pi: policy vector of size self.getActionSize()
        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass


class INeuralNet:
    def predict(self, input):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

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

    def load_checkpoint(self, folder, filename_no):
        """
        Loads parameters of the neural network from folder/filename
        """
    def save_checkpoint(self, folder, filename_no):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """


class NeuralNet(INeuralNet):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, action_size, version, args):
        self.args = args
        self.action_size = action_size
        self.board_x = 7
        self.board_y = 7

        # Neural Net - version with long form of board including status and number of pieces
        # self.input_boards = Input(shape=(4, self.board_x, self.board_y))  # s: batch_size x board_x x board_y
        # x_image = BatchNormalization(axis=3)(Reshape((self.board_x, self.board_y, 4))(self.input_boards))  # batch_size  x board_x x board_y x 4
        self.input_boards = Input(shape=(4, self.board_x, self.board_y))  # s: batch_size x board_x x board_y
        if version == 36:
            self.InitVersion36()
        else:
            self.InitVersion37()

    def InitVersion36(self):
        x_image = BatchNormalization(axis=3)(
            Reshape((self.board_x, self.board_y, 4))(self.input_boards))  # batch_size  x board_x x board_y x 4

        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 2)(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv11 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 1)(h_conv1)))  # batch_size  x board_x x board_y x num_channels

        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 2)(h_conv11)))  # batch_size  x board_x x board_y x num_channels
        h_conv21 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 1)(h_conv2)))  # batch_size  x board_x x board_y x num_channels

        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 2)(h_conv21)))  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv31 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 1)(h_conv3)))  # batch_size  x (board_x) x (board_y) x num_channels

        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 2)(h_conv31)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv41 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 1)(h_conv4)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv41)

        s_fc1 = Dropout(self.args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(self.args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           loss_weights=[1., 5.], optimizer=Adam(self.args.lr))
        print(self.model.summary())

    def InitVersion37(self):
        x_image = BatchNormalization(axis=3)(
            Reshape((self.board_x, self.board_y, 4))(self.input_boards))  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 3, padding='same')(
                x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(self.args.num_channels, 3, padding='same')(
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
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           loss_weights=[1., 5.], optimizer=Adam(self.args.lr))
        print(self.model.summary())

    def evaluate(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        a = self.model.evaluate(x=input_boards, y=[target_pis, target_vs], batch_size=self.args.batch_size, verbose=1)

        print(a)
        # print(b)

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

        plt.clf()
        epochs = range(1, len(result.history['loss']) + 1)
        plt.plot(epochs, result.history['loss'], 'bo', label='Training acc')
        plt.plot(epochs, result.history['val_loss'], 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.ion()
        plt.show()
        # plt.draw()
        plt.pause(0.001)

    def predict(self, input):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pi, v = self.model.predict(input)
        return pi[0], v[0]

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


# instance of a IGame
class Moara(IGame):
    # valid position where pieces reside
    VALID_POSITIONS = [(0, 0), (0, 3), (0, 6),
                       (1, 1), (1, 3), (1, 5),
                       (2, 2), (2, 3), (2, 4),
                       (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6),
                       (4, 2), (4, 3), (4, 4),
                       (5, 1), (5, 3), (5, 5),
                       (6, 0), (6, 3), (6, 6)]

    # transition from one position to another
    VALID_MOVES = [
        # move pieces on the table
        # horizontal
        ((0, 0), (0, 3)), ((0, 3), (0, 6)),
        ((0, 3), (0, 0)), ((0, 6), (0, 3)),

        ((1, 1), (1, 3)), ((1, 3), (1, 5)),
        ((1, 3), (1, 1)), ((1, 5), (1, 3)),

        ((2, 2), (2, 3)), ((2, 3), (2, 4)),
        ((2, 3), (2, 2)), ((2, 4), (2, 3)),

        ((3, 0), (3, 1)), ((3, 1), (3, 2)),
        ((3, 1), (3, 0)), ((3, 2), (3, 1)),

        ((3, 4), (3, 5)), ((3, 5), (3, 6)),
        ((3, 5), (3, 4)), ((3, 6), (3, 5)),

        ((4, 2), (4, 3)), ((4, 3), (4, 4)),
        ((4, 3), (4, 2)), ((4, 4), (4, 3)),

        ((5, 1), (5, 3)), ((5, 3), (5, 5)),
        ((5, 3), (5, 1)), ((5, 5), (5, 3)),

        ((6, 0), (6, 3)), ((6, 3), (6, 6)),
        ((6, 3), (6, 0)), ((6, 6), (6, 3)),
        # vertical
        ((0, 0), (3, 0)), ((3, 0), (6, 0)),
        ((3, 0), (0, 0)), ((6, 0), (3, 0)),

        ((1, 1), (3, 1)), ((3, 1), (5, 1)),
        ((3, 1), (1, 1)), ((5, 1), (3, 1)),

        ((2, 2), (3, 2)), ((3, 2), (4, 2)),
        ((3, 2), (2, 2)), ((4, 2), (3, 2)),

        ((0, 3), (1, 3)), ((1, 3), (2, 3)),
        ((1, 3), (0, 3)), ((2, 3), (1, 3)),

        ((4, 3), (5, 3)), ((5, 3), (6, 3)),
        ((5, 3), (4, 3)), ((6, 3), (5, 3)),

        ((2, 4), (3, 4)), ((3, 4), (4, 4)),
        ((3, 4), (2, 4)), ((4, 4), (3, 4)),

        ((1, 5), (3, 5)), ((3, 5), (5, 5)),
        ((3, 5), (1, 5)), ((5, 5), (3, 5)),

        ((0, 6), (3, 6)), ((3, 6), (6, 6)),
        ((3, 6), (0, 6)), ((6, 6), (3, 6))
    ]
    MILLS = [
        # horizontal
        ((0, 0), (0, 3), (0, 6)),
        ((1, 1), (1, 3), (1, 5)),
        ((2, 2), (2, 3), (2, 4)),
        ((3, 0), (3, 1), (3, 2)),
        ((3, 4), (3, 5), (3, 6)),
        ((4, 2), (4, 3), (4, 4)),
        ((5, 1), (5, 3), (5, 5)),
        ((6, 0), (6, 3), (6, 6)),
        # vertical
        ((0, 0), (3, 0), (6, 0)),
        ((1, 1), (3, 1), (5, 1)),
        ((2, 2), (3, 2), (4, 2)),
        ((0, 3), (1, 3), (2, 3)),
        ((4, 3), (5, 3), (6, 3)),
        ((2, 4), (3, 4), (4, 4)),
        ((1, 5), (3, 5), (5, 5)),
        ((0, 6), (3, 6), (6, 6))
    ]

    def __init__(self, ):
        self.playerAtMove = 1
        # history of positions
        self.history = {}
        # list of moves
        self.moves = []
        # memorize move results (s,a) -> s
        self.memo = {}
        # number of moves since last capture
        self.noMovesWithoutCapture = 0;
        # number of total moves
        self.noMoves = 0
        self.canonized = False  # true if canonical has revert the sign of the player
        self.n = 7

        """
                Returns:
                    startBoard: a representation of the board (ideally this is the form
                                that will be the input to your neural network)
                """
        # first plane - pieces
        arr1 = np.array([[0 for y in range(7)] for x in range(7)])
        # second plane - number of pieces for player 1
        arr2 = np.array([[0. for _ in range(7)] for _ in range(7)])
        arr2[3][3] = 9.0  # unused player 1 pieces
        # third plane - number of pieces for player 2
        arr3 = np.array([[0. for _ in range(7)] for _ in range(7)])
        arr3[3][3] = 9.0  # unused player 2 pieces
        # fourth plane - flag is current player must capture
        arr4 = np.array([[0. for _ in range(7)] for _ in range(7)])
        arr4[3][3] = 0.0
        self.internalArray = np.array([arr1, arr2, arr3, arr4])

    def reset(self):
        self.__init__()

    # return a copy of the game
    def copy(self):
        return copy.deepcopy(self)

    # add reward for capture
    def getExtraReward(self):
        return 0

    # string representation of the current position in the game
    def __repr__(self):
        hh = ''
        for (x, y) in Moara.VALID_POSITIONS:
            if self.getPosition((x, y)) == 1:
                hh += "x"
            elif self.getPosition((x, y)) == -1:
                hh += "o"
            else:
                hh += "_"
        hh = hh + " "
        hh = hh + str(self.getPlayerCount(1))  # player
        hh = hh + " "
        hh = hh + str(self.getPlayerCount(-1))  # opponent
        hh = hh + " "
        hh = hh + str(self.getBoardStatus())  # capture
        return hh

    def toShortString(self):
        hh = ''
        for (x, y) in Moara.VALID_POSITIONS:
            if self.getPosition((x, y)) == 1:
                hh += "x"
            elif self.getPosition((x, y)) == -1:
                hh += "o"
            else:
                hh += "_"
        return hh

    def display(self, invariant=1):
        n = self.internalArray.shape[1]
        print("")
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print(f"X={self.getPlayerCount(1)} Y={self.getPlayerCount(-1)}. Move #{self.noMoves}", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = self.getPosition((y, x)) * 1 * invariant  # get the piece to print
                if piece == 1:
                    print("X ", end="")
                elif piece == -1:
                    print("O ", end="")
                else:
                    if (y, x) in self.VALID_POSITIONS:
                        # if (str(self), Game.validPositions.index((y, x))) in mcts.Qsa:
                        if x == n:
                            print(".", end="")
                        else:
                            print(". ", end="")
                        # else:
                        #     if x == n:
                        #         print("!", end="")
                        #     else:
                        #         print("! ", end="")
                    else:
                        if x == 3 and y == 3 and self.getBoardStatus() != 0:
                            if np.sign(self.getBoardStatus()) == 1:
                                print(chr(96 + abs(self.getBoardStatus())), end=" ")
                            else:
                                print(chr(96 + abs(self.getBoardStatus())), end="-")
                        elif x == n:
                            print(" ", end="")
                        else:
                            print("  ", end="")

            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")

    def getCanonicalForm(self):

        if self.playerAtMove == 1:
            self.canonized = False
            return self.copy()
        else:

            temp = self.copy()
            temp.canonized = True
            temp.internalArray = np.array(
                [self.internalArray[0] * self.playerAtMove, self.internalArray[2], self.internalArray[1],
                 self.internalArray[3] * self.playerAtMove])
            temp.playerAtMove = 1
            # also revert for history as well
            return temp

    def getCrtPlayer(self) -> int:
        return self.playerAtMove

    def getPosition(self, pos):
        (x, y) = pos
        return self.internalArray[0][y][x]

    def setPosition(self, pos, value):
        (x, y) = pos
        self.internalArray[0][y][x] = value

    def getBoardStatus(self):
        return int(round(self.internalArray[3][3][3]))

    def setBoardStatus(self, value):
        self.internalArray[3][3][3] = value

    # total number of pieces for the player
    def getPlayerCount(self, player):
        no_pieces_on_board = len([0 for pos in self.VALID_POSITIONS if self.getPosition(pos) == player])
        return no_pieces_on_board + self.getUnusedPlayerCount(player)

    # number of unused pieces for the player
    def getUnusedPlayerCount(self, player):
        if player == 1:
            return int(self.internalArray[1][3][3])
        else:
            return int(self.internalArray[2][3][3])

    def decUnusedPlayerCount(self, player):
        if player == 1:
            self.internalArray[1][3][3] = int(self.internalArray[1][3][3]) - 1
        else:
            self.internalArray[2][3][3] = int(self.internalArray[2][3][3]) - 1

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return (24 +  # fly 1st piece
                #         24 +  # fly 2nd piece
                #         24 +  # fly 3rd piece
                #         24 +  # put piece
                #         24 +  # capture piece
                len(self.VALID_MOVES) +  # move piece
                1)  # pass

    def isMill(self, mill, player):
        count = functools.reduce(lambda acc, i: acc + (1 if self.getPosition(mill[i]) == player else 0),
                                 range(3), 0)
        if count == 3:
            return 1
        else:
            return 0

    def isInAMill(self, pos, player):
        # find all mills that contain the pos
        mill_list = list(filter(lambda mill: list(filter(lambda p: p == pos, mill)) != [], self.MILLS))
        return list(filter(lambda x: self.isMill(x, player) == 1, mill_list)) != []

    # list of legal moves from the current position for the player
    def getValidMoves(self, player):
        boardStatus = self.getBoardStatus()
        s = self.toShortString()
        # no more than 3 repetitions allowed
        if s in self.history and self.history[s] > 2:
            return []
        # no more than 50 moves without capture
        if self.noMovesWithoutCapture > 50:
            return []
        result = []
        # normal operations
        if boardStatus == 0:
            if self.getUnusedPlayerCount(player) > 0:  # put
                # phase 1: can put anywhere where there is an empty place
                result = list(filter(lambda x: self.getPosition(x) == 0, self.VALID_POSITIONS))
                result = [self.VALID_POSITIONS.index(x) for x in result]
            elif self.getPlayerCount(player) > 3:  # move
                # phase 2: need to move a piece in a valid position
                # select all transitions that have a player piece in the first position and an empty one in the second position
                result = list(filter(lambda x: self.getPosition(x[0]) == player and
                                               self.getPosition(x[1]) == 0, self.VALID_MOVES))
                # result = [x[0] for x in result]
                result = [24 + self.VALID_MOVES.index(x) for x in result]
            elif self.getPlayerCount(player) == 3:  # jump
                # phase 3: when only 3 pieces left can move anywhere empty, select any piece
                result = list(filter(lambda x: self.getPosition(x) == player, self.VALID_POSITIONS))
                result = [self.VALID_POSITIONS.index(x) for x in result]
            else:
                assert (False)  # should never come here, it's lost
        else:
            # special case, when a capture must be chosen or a jump when <= 3 pieces remaining
            if boardStatus != 0 and np.sign(boardStatus) != np.sign(player):
                result = [24 + len(self.VALID_MOVES)]  # pass
            else:
                # if the player has a mill, it must remove an opponent's piece,
                if boardStatus == 100 * player:
                    available_opponent_pieces = list(
                        filter(lambda x: self.getPosition(x) == -player, self.VALID_POSITIONS))
                    # that is not an enemy mill
                    result = list(
                        filter(lambda p: self.isInAMill(p, -player) is False, available_opponent_pieces))
                    result = [self.VALID_POSITIONS.index(x) for x in result]
                    if len(result) == 0:
                        self.display()
                        # if no available enemy piece to capture outside of aa mill
                        # retry with a piece from an enemy mill
                        result = [self.VALID_POSITIONS.index(x) for x in available_opponent_pieces]
                else:  # select where to jump
                    result = list(filter(lambda x: self.getPosition(x) == 0, self.VALID_POSITIONS))
                    result = [self.VALID_POSITIONS.index(x) for x in result]
        # simulate each move, so that it wouldn't get into a invalid condition
        # 3 state repetition or 50 moves without capture
        # possibleMoves = []
        # for move in result:
        #     if self.getBoardStatus() == 0:
        #         # if (s, move) not in self.memo:
        #         newState = self.getNextState(move)
        #         s = newState.toShortString()
        #         if self.canonized:
        #             # transform s
        #             l = list(s)
        #             l = ['x' if x == 'o' else 'o' if x == 'x' else '_' for x in l]
        #             s = ''.join(l)
        #         if newState.history[s] < 3 and newState.noMovesWithoutCapture < 50:
        #             possibleMoves.append(move)
        #     else:
        #         possibleMoves.append(move)

        return result

    def getNextState(self, action):
        # state will be modified in the copied object
        newGameState = self.copy()
        newGameState.playerAtMove = -self.playerAtMove
        boardStatus = self.getBoardStatus()
        # if player must make a move or capture, i.e. board status != 0,
        # the opponent takes a dummy move, nothing changes
        if action == 24 + len(self.VALID_MOVES):  # pass
            return newGameState

        # capture
        if abs(boardStatus) == 100:  # capture
            # newGameState.display()
            newPosition = self.VALID_POSITIONS[action]
            assert (self.getPosition(newPosition) == -self.playerAtMove)
            newGameState.setPosition(newPosition, 0)
            newGameState.setBoardStatus(0)
            newGameState.noMovesWithoutCapture = 0  # reset it
            # newGameState.display()
        elif boardStatus == 0:  # select/put piece
            player_no = self.getPlayerCount(self.playerAtMove)
            if action < len(self.VALID_POSITIONS):  # put or jump type of action
                if self.getUnusedPlayerCount(self.playerAtMove) > 0:  # put
                    newPosition = self.VALID_POSITIONS[action]
                    newGameState.setPosition(newPosition, self.playerAtMove)
                    newGameState.decUnusedPlayerCount(self.playerAtMove)
                else:  # jump
                    assert (player_no == 3)
                    old = self.VALID_POSITIONS[action]
                    assert (self.getPosition(old) == self.playerAtMove)
                    newGameState.setBoardStatus((action + 1) * self.playerAtMove)
            else:  # move type of action
                if player_no > 3:
                    move = self.VALID_MOVES[action - 24]
                    old = move[0]  # from
                    assert (self.getPosition(old) == self.playerAtMove)
                    newGameState.setPosition(old, 0)
                    newPosition = move[1]  # to
                    assert (self.getPosition(newPosition) == 0)
                    newGameState.setPosition(newPosition, self.playerAtMove)
                else:
                    assert (False)
        elif boardStatus != 0:  # select where to jump to
            old = self.VALID_POSITIONS[abs(boardStatus) - 1]
            newPosition = self.VALID_POSITIONS[action]
            # make sure we start from player
            if boardStatus != self.playerAtMove:
                print(f"not player at {old}")
                self.display()
            assert (boardStatus == self.playerAtMove)
            # make sure it's empty
            if self.getPosition(newPosition) != 0:
                print(f"not empty for {newPosition}")
                self.display()
            assert (self.getPosition(newPosition) == 0)
            newGameState.setPosition(old, 0)
            newGameState.setPosition(newPosition, self.playerAtMove)
            newGameState.setBoardStatus(0)
        if boardStatus == 0:
            if newGameState.isInAMill(newPosition, self.playerAtMove):
                newGameState.setBoardStatus(100 * self.playerAtMove)  # flag that a capture can be made
        newGameState.updateHistory()
        return newGameState

    def updateHistory(self):
        s = self.toShortString()
        if self.canonized:
            # transform s
            l = list(s)
            l = ['x' if x == 'o' else 'o' if x == 'x' else '_' for x in l]
            s = ''.join(l)
        if s not in self.history:
            self.history[s] = 1
        else:
            self.history[s] += 1
        self.moves.append(s)
        self.noMoves += 1
        self.noMovesWithoutCapture += 1

    def getGameEnded(self) -> float:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player win if:
        #   - opponent has less than 3 pieces
        #   - opponent has no valid moves
        # draw if:
        #   - no valid moves for any player
        #   - future:
        #       - 50 moves with no capture
        #       - position replay 3 times
        player = self.playerAtMove
        player_valid_moves_list = self.getValidMoves(player)
        opponent_valid_moves_list = self.getValidMoves(-player)
        if player_valid_moves_list == [] and opponent_valid_moves_list == []:
            return 0.00000000001  # draw
        if self.getPlayerCount(player) < 3 or player_valid_moves_list == []:
            return -1
        if self.getPlayerCount(-player) < 3 or opponent_valid_moves_list == []:
            return 1

        return 0

    def getInternalRepresentation(self):
        '''
        self.playerAtMove = 1
        # first plane - pieces
        arr1 = np.array([[0 for y in range(7)] for x in range(7)])
        # second plane - number of pieces for player 1
        arr2 = np.array([[0. for _ in range(7)] for _ in range(7)])
        arr2[3][3] = 9.0  # unused player 1 pieces
        # third plane - number of pieces for player 2
        arr3 = np.array([[0. for _ in range(7)] for _ in range(7)])
        arr3[3][3] = 9.0  # unused player 2 pieces
        # fourth plane - flag is current player must capture
        arr4 = np.array([[0. for _ in range(7)] for _ in range(7)])
        arr4[3][3] = 0.0
        self.internalArray = np.array([arr1, arr2, arr3, arr4])
        '''
        return self.internalArray[np.newaxis, :, :]

    def getSymmetries(self, pi):
        """
        Input:
            pi: policy vector of size self.getActionSize()
        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [pi]


class MCTS:
    def __init__(self, nnet: NeuralNet):
        self.Quality = {}  # quality of taken action a from state s
        self.NumberOfActionTaken = {}  # stores #times action a was taken from board s
        self.NumberOfVisits = {}  # stores #times board s was encountered in the mcts search
        self.Prediction = {}  # prediction of taken action a from state s (returned by neural net)
        self.nnet = nnet

        #how many times an existing node was traversed
        self.ExistingNodesCounter = 0
        # how many times a new node was created
        self.NewNodesCounter = 0

    def reset(self):
        self.Quality = {}  # quality of taken action a from state s
        self.NumberOfActionTaken = {}  # stores #times action a was taken from board s
        self.NumberOfVisits = {}  # stores #times board s was encountered in the mcts search
        self.Prediction = {}  # prediction of taken action a from state s (returned by neural net)

        # how many times an existing node was traversed
        self.ExistingNodesCounter = 0
        # how many times a new node was created
        self.NewNodesCounter = 0

    def iterateNode(self, game: IGame):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        endGameScore = game.getGameEnded()
        # game has ended, it's a terminal node
        if endGameScore != 0:
            return -endGameScore
        # it's not a terminal node, continue selecting/creating a sub-node
        s = str(game)
        if s not in self.Prediction:
            inputToNN = game.getInternalRepresentation()
            policy_predicted, value = self.nnet.predict(inputToNN)
            validMoves = game.getValidMoves(game.getCrtPlayer())
            moves = [1 if x in validMoves else 0 for x in range(game.getActionSize())]
            policy = policy_predicted * moves  # masking invalid moves
            sum_Policies = np.sum(policy)
            if sum_Policies > 0:
                policy /= sum_Policies  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                policy = policy + moves
                if np.sum(policy) != 0:
                    policy /= np.sum(policy)
            self.Prediction[s] = policy
            self.NumberOfVisits[s] = 0
            self.NewNodesCounter += 1
            # print(".", end=" ")
            return -value[0] #+ game.getExtraReward()

        self.ExistingNodesCounter += 1
        a = self.getBestAction(game)
        newg = game.getNextState(a).getCanonicalForm()
        # add a reward for capture
        v = newg.getExtraReward()
        v += self.iterateNode(newg)
        if (s, a) in self.Quality:
            q = self.Quality[(s, a)]
            na = self.NumberOfActionTaken[(s, a)]
            q = (na * q + v) / (na + 1)
            self.Quality[(s, a)] = q
            self.NumberOfActionTaken[(s, a)] = na + 1

        else:
            self.Quality[(s, a)] = v
            self.NumberOfActionTaken[(s, a)] = 1
        n = self.NumberOfVisits[s] + 1
        self.NumberOfVisits[s] = n
        return -v

    def getBestAction(self, game: IGame):
        EPS = 1e-8
        s = str(game)
        # cur_best = -float('inf')
        # best_act = -1
        # # UCT calculation
        # for a in game.getValidMoves(game.getCrtPlayer()):
        #     if (s, a) in self.Quality:
        #         q = self.Quality[(s, a)]
        #         p = self.Prediction[s][a]
        #         n = self.NumberOfVisits[s]
        #         na = self.NumberOfActionTaken[(s, a)]
        #         u = q + moara.args.cpuct * p * math.sqrt(n) / (1 + na)
        #     else:
        #         p_ = self.Prediction[s][a]
        #         n_ = self.NumberOfVisits[s]
        #         u = moara.args.cpuct * p_ * math.sqrt(n_ + EPS)  # Q = 0 ?
        #
        #     if u > cur_best:
        #         cur_best = u
        #         best_act = a
        # a = best_act

        #compute list of u values, functionally
        valid_actions = game.getValidMoves(game.getCrtPlayer())
        u_values = [
            (self.Quality[(s, a)] + moara.args.cpuct * self.Prediction[s][a] * math.sqrt(self.NumberOfVisits[s]) / (1 + self.NumberOfActionTaken[(s, a)]))
            if (s, a) in self.Quality
            else (moara.args.cpuct * self.Prediction[s][a] * math.sqrt(self.NumberOfVisits[s] + EPS)) for a in valid_actions]
        # get action with the best ucb, depending on current player
        # best_pair = functools.reduce(lambda acc, pair: pair if pair[1] > acc[1] else acc, zip(valid_actions, u_values))
        if game.getCrtPlayer() == 1:#if first player, choose max value
            best_pair = functools.reduce(lambda acc, pair: pair if pair[1] > acc[1] else acc, zip(valid_actions, u_values))
        else:#if second player, choose min value
            best_pair = functools.reduce(lambda acc, pair: pair if pair[1] < acc[1] else acc, zip(valid_actions, u_values))

        a = best_pair[0]
        return a

    def getActionProbabilities(self, game: IGame, temperature: float = 1, simulate: bool = True):
        """
                This function performs numMCTSSims simulations of MCTS starting from
                current game position.

                Returns:
                    probs: a policy vector with the probability of each action
                           proportional to Nsa[(s,a)]**(1./temp)
        """
        if simulate:
            for i in range(moara.args.numMCTSSimulations):
                # simulate the game a move at a time
                v = self.iterateNode(game)

        s = str(game)
        actionCounters = [self.NumberOfActionTaken[(s, a)] if (s, a) in self.NumberOfActionTaken else 0 for a in
                          range(game.getActionSize())]

        if temperature == 0:
            bestA = np.argmax(actionCounters)
            probabilities = [0] * len(actionCounters)
            probabilities[bestA] = 1
        else:
            counts = [x ** (1. / temperature) for x in actionCounters]
            # normalization
            sumcounts = float(sum(counts))
            probabilities = [x / sumcounts for x in actionCounters]
        return probabilities


def executeEpisode(game: IGame, mcts: MCTS):
    trainExamples = []
    game.reset()
    episodeStep = 0

    # loop until game ends
    while True:
        episodeStep += 1
        canonical = game.getCanonicalForm()
        temperature: int = 1 #int(episodeStep < moara.args.tempThreshold)

        probabilities = mcts.getActionProbabilities(canonical, temperature)
        # sym = canonical.getSymmetries(probabilities)

        action = np.random.choice(len(probabilities), p=probabilities)
        # for p in sym:
        trainExamples.append([canonical.getInternalRepresentation(), game.getCrtPlayer(), probabilities])

        game:IGame = game.getNextState(action)
        # game.display()
        s = str(canonical)
        if (s, action) in mcts.Quality:
            # print(str(episodeStep) + ": " + str(self.mcts.Qsa[(s, action)]) + " - " + str(board))
            print(".", end="", flush=True)
        mcts.ExistingNodesCounter = 0
        mcts.NewNodesCounter = 0
        r = game.getGameEnded()

        if r != 0:
            if abs(r) < 1:
                print("")
                print(f"{episodeStep:003d}: {mcts.Quality[(s, action)] * (-game.playerAtMove):+4.2f} : {action:4d} : {game} {game.getPlayerCount(1)}-{game.getPlayerCount(-1)} ex:{mcts.ExistingNodesCounter} new:{mcts.NewNodesCounter}")
                print(f"Game ended after {game.noMovesWithoutCapture} moves")
            else:
                print("")
                print(f"{episodeStep:003d}: {mcts.Quality[(s, action)] * (-game.playerAtMove):+4.2f} : {action:4d} : {game} {game.getPlayerCount(1)}-{game.getPlayerCount(-1)} ex:{mcts.ExistingNodesCounter} new:{mcts.NewNodesCounter}")
            game.SaveData()
            return [(x[0], x[2], r * ((-1) ** (x[1] != game.getCrtPlayer()))) for x in trainExamples]


def learn(game: IGame, mcts:MCTS, nnet: INeuralNet, doArena = None):
    for i in range(0, moara.args.numIterations + 1):
        iterationTrainExamples = deque([], maxlen=moara.args.maxlenOfQueue)
        print('------ITER ' + str(i) + '------')
        for episode in range(moara.args.numEpisodes):
            print(f"----- Episode {episode} -----")
            mcts.reset()
            example = executeEpisode(game, mcts)
            if example != []:
                iterationTrainExamples += example
        trainExamplesHistory.append(iterationTrainExamples)
        print(f"len(trainExamplesHistory) ={len(trainExamplesHistory)}")
        if len(trainExamplesHistory) > moara.args.numItersForTrainExamplesHistory:
            print("len(trainExamplesHistory) =", len(trainExamplesHistory),
                  " => remove the oldest trainExamples")
            trainExamplesHistory.pop(0)

        # shuffle examlpes before training
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)
        if trainExamples != []:
            nnet.train(trainExamples)
            nnet.save_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)
            # test against the randome
            # if i % 5 == 0:
            if doArena is not None:
                doArena(nnet, False)


# def PitAgainst(neuralDataFileNumber):
#
#     # self.pnet = self.nnet.__class__(self.game, args)  # the competitor network
#     if self.pnet is None:
#         self.pnet = NeuralNet(g, 37, moara.args)
#     self.pnet.load_checkpoint(folder=self.args.checkpoint, filename_no=neuralDataFileNumber)
#     pmcts = MCTS(self.game, self.pnet, self.args)
#     nmcts = MCTS(self.game, self.nnet, self.args)
#
#     print(f'PITTING AGAINST {neuralDataFileNumber}')
#     arena = moara.Arena(lambda x: np.argmax(nmcts.getActionProb(x, 0)),
#                   lambda x: np.argmax(pmcts.getActionProb(x, 0)), self.game, self.args)
#     nwins, pwins, draws = arena.playGames(self.args.arenaCompare, verbose=True)
#     print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

if __name__ == "__main__":
    print("mcts 2")
    moaraGame: Moara = Moara()
    n = NeuralNet(moaraGame.getActionSize(), 0, moara.args)
    n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)




    mcts = MCTS(n)
    learn(moaraGame, mcts, n, None)

