import functools
import math
import os

import keras
import numpy as np
from keras import Input, Model
from keras.layers import BatchNormalization, Reshape, Activation, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import moara
import copy


# interface for game classes
class IGame:
    pass
class IGame:
    def InitializeBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getGameEnded(self, canonical: bool = True) -> float:
        """
        Input:
            canonical: whether to use the cannonical form, i.e. player invariant

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def GetInternalRepresentation(self):
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

    def getNextState(self, player, action) -> IGame:
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def getCanonicalForm(self) -> IGame:
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


class NeuralNet:
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
                       (3, 0), (3, 1), (3, 2),
                       (3, 4), (3, 5), (3, 6),
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
        self.internalArray = []
        # history of positions
        self.history = {}
        # number of moves since last capture
        self.noMovesWithoutCapture = 0;
        # number of total moves
        self.noMoves = 0
        self.InitializeBoard()

    # return a copy of the game
    def copy(self):
        return copy.deepcopy(self)

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

    def getCanonicalForm(self):
        if self.playerAtMove == 1:
            return self.copy()
        else:
            temp = self.copy()
            temp.internalArray = np.array(
                [self.internalArray[0] * self.playerAtMove, self.internalArray[2], self.internalArray[1],
                 self.internalArray[3] * self.playerAtMove])
            return temp

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

    def InitializeBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # first plane - pieces
        arr1 = np.array([[0 for y in range(7)] for x in range(7)])
        # arr1[0][0] = 1
        # arr1[3][0] = 1
        # arr1[6][0] = 1
        # arr1[4][4] = 1
        # arr1[6][6] = 1
        # arr1[2][2] = -1
        # arr1[1][1] = -1
        # arr1[3][1] = -1
        # arr1[5][1] = -1
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

        # normal operations
        if boardStatus == 0:
            # phase 1: can put anywhere where there is an empty place
            if self.getUnusedPlayerCount(player) > 0:

                result = list(filter(lambda x: self.getPosition(x) == 0, self.VALID_POSITIONS))
                return [self.VALID_POSITIONS.index(x) for x in result]
            else:
                pass
        else:
            # special case, when a capture must be chosen or a jump when <= 3 pieces remaining
            pass
        return []

        # # opposite player passes
        # if boardStatus != 0 and np.sign(boardStatus) != np.sign(player):
        #     return [24 + len(self.VALID_MOVES)]  # pass
        #
        # # only if the last move results in a mill
        # # if the player has a mill, it must remove an opponent's piece, that is not a mill either
        # # mill_no = functools.reduce(lambda acc, mill: acc + isMill(board, mill, player), mills, 0)
        # # need to select an opponent piece
        # if boardStatus == 100 * player:
        #     available_opponent_pieces = list(filter(lambda x: self.getPosition(x) == -player, self.VALID_POSITIONS))
        #     result = list(filter(lambda p: self.isInAMill(board, p, -player) is False, available_opponent_pieces))
        #     result = [self.VALID_POSITIONS.index(x) for x in result]
        #     if len(result) > 0:
        #         return result  # else choose another move
        #     else:
        #         # print("can't capture")
        #         # board.setBoardStatus(0)
        #         boardStatus = 0
        # # move piece, select destination, phase 2 or 3
        #
        # # pieces_on_board = functools.reduce(lambda acc, pos: acc + (1 if self.getPosition(board, pos) == player else 0),
        # #                                    self.validPositions, 0)
        # pieces_on_board = len([0 for pos in self.VALID_POSITIONS if board.getPosition(pos) == player])
        # result = []
        #
        # player_no = board.getPlayerCount(player)
        #
        # if boardStatus != 0:
        #     # select those actions that originate in the stored position
        #     boardStatus = boardStatus * player
        #     if player_no > 3:  # move
        #         # (x, y) = Game.validPositions[boardStatus - 1]
        #         # result = list(filter(lambda a: a[0] == (x, y), Game.validActions))
        #         # result = [x[1] for x in result if board.getPosition(x[1]) == 0]
        #         # result = set(result)
        #         # result = [Game.validPositions.index(x) for x in result]
        #         result = list(filter(lambda x: board.getPosition(x[0]) == player and
        #                                        board.getPosition(x[1]) == 0,
        #                              self.validActions))
        #         result = [24 + self.validPositions.index(x) for x in result]
        #     if player_no == 3:
        #         # any empty place
        #         result = list(filter(lambda x: board.getPosition(x) == 0, Game.validPositions))
        #         result = [Game.validPositions.index(x) for x in result]
        #     return result
        # if player_no > pieces_on_board:  # there are still pieces to put on board
        #     # phase 1: can put anywhere where there is an empty place
        #     result = list(filter(lambda x: board.getPosition(x) == 0, Game.validPositions))
        #     # result = [3 * 24 + self.validPositions.index(x) for x in result]
        #     result = [Game.validPositions.index(x) for x in result]
        # elif player_no > 3:
        #     # phase 2: need to move a piece in a valid position
        #     # select all transitions that have a player piece in the first position and an empty one in the second position
        #     result = list(filter(lambda x: board.getPosition(x[0]) == player and
        #                                    board.getPosition(x[1]) == 0, Game.validActions))
        #     # result = [x[0] for x in result]
        #     # result = set(result)
        #     result = [24 + Game.validActions.index(x) for x in result]
        # elif player_no == 3:
        #     # phase 3: when only 3 pieces left can move anywhere empty
        #     result = list(filter(lambda x: board.getPosition(x) == player, Game.validPositions))
        #     result = [Game.validPositions.index(x) for x in result]
        # return result

    def getNextState(self, player, action):
        newGameState = self.copy()
        newGameState.playerAtMove = -player
        # if player takes action on board, return next (board,player)

        # if player must make a move or capture, i.e. board status != 0,
        # the opponent takes a dummy move, nothing changes
        if action == 24 + len(self.VALID_MOVES):  # pass
            newGameState.UpdateHistory()
            return newGameState

        # action must be a valid move
        # pos = (3, 3)
        # pieces_on_board = len([0 for pos in Game.validPositions if board.getPosition(pos) == player])
        # player_no = board.getPlayerCount(player)

        # pre-selection for capture
        if abs(newGameState.getBoardStatus()) == 100:  # capture
            # could not capture, but can move
            if action >= 24:
                newGameState.setBoardStatus(0)
            # could not capture, but can jump
            else:
                move = self.VALID_POSITIONS[action]
                if newGameState.getPosition(move) != -player:
                    newGameState.setBoardStatus(0)

        # phase 1
        if newGameState.getBoardStatus() == 0:  # select/put piece
            if action < len(self.VALID_POSITIONS):  # put
                pos = self.VALID_POSITIONS[action]
                newGameState.setPosition(pos, player)
                newGameState.decUnusedPlayerCount(player)
            else:  # prepare move

                if player_no > 3:
                    move = Game.validActions[action - 24]
                    pos = move[0]  # from
                    if board.getPosition(pos) != player:
                        board.display(player)
                    assert (board.getPosition(pos) == player)
                    board.setPosition(pos, 0)
                    pos = move[1]  # to
                    if board.getPosition(pos) != 0:
                        board.display(player)
                    assert (board.getPosition(pos) == 0)
                    board.setPosition(pos, player)
                elif player_no == 3:
                    orig = Game.validPositions[action]
                    assert (board.getPosition(orig) == player)
                    board.setBoardStatus((action + 1) * player)
                else:
                    assert (False)
        elif abs(board.getBoardStatus()) == 100:  # capture
            # make sure flag is used only once
            # double check
            # remove piece
            pos = Game.validPositions[action]
            if board.getPosition(pos) != -player:
                aaaa = 0
            assert (board.getPosition(pos) == -player)
            board.setPosition(pos, 0)
            board.setOpponentCount(player, board.getOpponentCount(player) - 1)

            board.setBoardStatus(0)
            pass
        elif board.getBoardStatus() != 0:  # move
            orig = Game.validPositions[abs(board.getBoardStatus()) - 1]
            pos = Game.validPositions[action]
            # make sure we start from player
            if board.getPosition(orig) != player:
                print(f"not player at {pos}")
                print(str(board))
                board.display(player)
                aaa = 3
            assert (board.getPosition(orig) == player)
            # make sure it's empty
            if board.getPosition(pos) != 0:
                print(f"not empty for {pos}")
                print(str(board))
                board.display(player)
                aaa = 3
            assert (board.getPosition(pos) == 0)

            board.setPosition(orig, 0)
            board.setPosition(pos, player)
            # make sure flag is used only once
            board.setBoardStatus(0)
        if newGameState.isInAMill(pos, player):
            newGameState.setBoardStatus(100 * player)  # flag that a capture can be made
        newGameState.UpdateHistory()
        return newGameState

    def UpdateHistory(self):
        s = str(self)
        if s not in self.history:
            self.history[s] = 1
        else:
            self.history[s] += 1

    def getGameEnded(self, canonical: bool = True) -> float:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player win if:
        #   - opponent has less than 3 pieces
        #   - opponent has no valid moves
        # draw if:
        #   - no valid moves for any player
        #   - future:
        #       - 50 moves with no capture
        #       - position replay 3 times
        player = 1 if canonical else self.playerAtMove
        player_valid_moves_list = self.getValidMoves(player)
        opponent_valid_moves_list = self.getValidMoves(-player)
        if player_valid_moves_list == [] and opponent_valid_moves_list == []:
            return 0.001  # draw
        if self.getPlayerCount(-player) < 3 or opponent_valid_moves_list == []:
            return 1
        if self.getPlayerCount(player) < 3 or player_valid_moves_list == []:
            return -1
        return 0

    def GetInternalRepresentation(self):
        return self.internalArray[np.newaxis, :, :]


class MCTS:
    def __init__(self, nnet: NeuralNet):
        self.Quality = {}  # quality of taken action a from state s
        self.NumberOfActionTaken = {}  # stores #times action a was taken from board s
        self.NumberOfVisits = {}  # stores #times board s was encountered in the mcts search
        self.Prediction = {}  # prediction of taken action a from state s (returned by neural net)
        self.nnet = nnet
        self.ValidMoves = {}  # stores game.getValidMoves for board s

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
            inputToNN = game.GetInternalRepresentation()
            policy, value = self.nnet.predict(inputToNN)
            validMoves = game.getValidMoves(1)
            moves = [1 if x in validMoves else 0 for x in range(game.getActionSize())]
            policy = policy * moves  # masking invalid moves
            sum_Policies = np.sum(policy)
            if sum_Policies > 0:
                policy /= sum_Policies  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                policy = policy + moves
                policy /= np.sum(policy)
            self.Prediction[s] = policy
            self.ValidMoves[s] = validMoves
            self.NumberOfVisits[s] = 0

            # print(".", end=" ")
            return -value[0]  # because value returned by nn is an array of 1

        a = self.getBestAction(s)
        game = game.getNextState(1, a).getCanonicalForm()
        v = self.iterateNode(game)
        if (s, a) in self.Quality:
            q = self.Quality[(s, a)]
            na = self.NumberOfActionTaken[(s, a)]
            q = (na * q + v) / (na + 1)
            self.Quality[(s, a)] = q
            self.NumberOfActionTaken[(s, a)] = na + 1

        else:
            self.Quality[(s, a)] = v
            self.NumberOfActionTaken[(s, a)] = 1
        self.NumberOfVisits[s] += 1
        return -v

    def getBestAction(self, s):
        EPS = 1e-8
        cur_best = -float('inf')
        best_act = -1
        # u_values = [
        #     (self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)]))
        #     if (s, a) in self.Qsa
        #     else (self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)) for a in valid_actions]
        # # get action with the best ucb
        # best_pair = functools.reduce(lambda acc, pair: pair
        # if pair[1] > acc[1] else acc, zip(valid_actions, u_values), (best_act, cur_best))
        # a = best_pair[0]

        # UCT calculation

        for a in self.ValidMoves[s]:
            if (s, a) in self.Quality:
                q = self.Quality[(s, a)]
                p = self.Prediction[s][a]
                n = self.NumberOfVisits[s]
                na = self.NumberOfActionTaken[(s, a)]
                u = q + moara.args.cpuct * p * math.sqrt(n) / (1 + na)
            else:
                p = self.Prediction[s][a]
                n = self.NumberOfVisits[s]
                u = moara.args.cpuct * p * math.sqrt(n + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a
        a = best_act
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
        counts = [self.NumberOfActionTaken[(s, a)] if (s, a) in self.NumberOfActionTaken else 0 for a in range(game.getActionSize())]

        return counts


def executeEpisode(game: IGame, mcts: MCTS):
    currentPlayer = 1  # alternate between 1(white) and -1(black)
    episodeStep = 0
    while True:
        episodeStep += 1
        temperature: int = int(episodeStep < moara.args.tempThreshold)
        mcts.getActionProbabilities(game, temperature)
        pass


def learn(game: IGame, mcts: MCTS):
    for i in range(0, moara.args.numIterations + 1):
        print('------ITER ' + str(i) + '------')
        for episode in range(moara.args.numEpisodes):
            print(f"----- Episode {episode} -----")
            executeEpisode(game, mcts)


print("mcts 2")
moaraGame: Moara = Moara()
n = NeuralNet(moaraGame.getActionSize(), 0, moara.args)

mcts = MCTS(n)
learn(moaraGame, mcts)
