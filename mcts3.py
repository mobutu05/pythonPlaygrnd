import copy
import functools
import os
import pickle

import numpy as np
from keras import Input, Model
from keras.layers import BatchNormalization, Reshape, Activation, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam

import mcts2
import moara
import matplotlib.pyplot as plt

class MoaraNew:
    pass


class NeuralNetNew(mcts2.INeuralNet):
    def __init__(self, game: MoaraNew, args):
        self.args = args
        self.action_size = game.getActionSize()
        self.board_x = game.board_X  # lines
        self.board_y = game.board_Y  # columns
        self.planes = (8 * (1 +  # board for player 1
                            1 +  # board for player 2
                            1 +  # rotated board for player 1
                            1 +  # rotated board for player 2
                            1 +  # unused pieces for player 1
                            1) +  # unused pieces for player 2
                       1 +  # no repetitions for current board, uniform value over array
                       1 +  # total moves, uniform value over array
                       1)  # moves without capture, uniform value over array

        self.input_boards = Input(shape=(self.planes, self.board_x, self.board_y))  # s: batch_size x board_x x board_y

        x_image = BatchNormalization(axis=3)(
            Reshape((self.board_x, self.board_y, self.planes))(
                self.input_boards))  # batch_size  x board_x x board_y x 1
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
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(
            s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           loss_weights=[1., 5.], optimizer=Adam(self.args.lr))
        print(self.model.summary())

    def predict(self, inputData):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
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


class MoaraNew(mcts2.IGame):
    # transition from one position to another
    VALID_MOVES = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.VALID_MOVES]
    MILLS = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.MILLS]
    # list of possible moves from each board configuration
    # this is independent of a particular game, so can be serialized
    ValidMovesFromState = {}

    def LoadValidMoves(self):
        folder = moara.args.checkpoint
        filename = f"valid.moves.data"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            infile = open(filepath, 'rb')
            MoaraNew.ValidMovesFromState = pickle.load(infile)
            infile.close()
        else:
            print("No model in path '{}'".format(filepath))
            ValidMovesFromState = {}

    def SaveValidMoves(self):
        folder = moara.args.checkpoint
        filename = f"valid.moves.data"
        filepath = os.path.join(folder, filename)
        outfile = open(filepath, 'wb')
        pickle.dump(MoaraNew.ValidMovesFromState, outfile)
        outfile.close()

    def __init__(self, ):
        self.board_X = 8
        self.board_Y = 3
        self.boardSize = self.board_X * self.board_Y
        self.unusedPieces = [9, 9]  # for opponent, player
        self.playerAtMove = 1
        self.internalArray = np.array([0 for _ in range(self.boardSize)])
        # history of positions
        self.history = {}
        # list of moves
        self.moves = []
        # number of moves since last capture
        self.noMovesWithoutCapture = 0
        # number of total moves
        self.noMoves = 0
        self.canonized = False  # true if canonical has revert the sign of the player
        self.time_len = 8  # keep current state and previous 7
        self.feature_len = 6  # number of planes describing current board
        self.timePlanes = [[[0 for _ in range(self.boardSize)] for _ in range(self.feature_len)] for _ in
                           range(self.time_len)]
        self.possibleMovesSize = (self.boardSize +  # put pieces (while unused pieces exist)
                                  self.boardSize * self.boardSize)  # move/jump pieces

    def SaveData(self):
        self.SaveValidMoves()

    def reset(self):
        self.__init__()

    def toShortString(self):
        return ''.join(['x' if x == 1 else 'o' if x == -1 else '_' for x in self.internalArray])

    # string representation of the current position in the game
    def __repr__(self):
        return f"{self.toShortString()}_{self.getUnusedPlayerCount(1)}_{self.getUnusedPlayerCount(-1)}"

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return (self.possibleMovesSize  # put pieces (while unused pieces exist) or move/jump pieces
                * 10)  # capture none or any of 9 opponent pieces

    def copy(self):
        return copy.deepcopy(self)

    def getCanonicalForm(self):
        if self.playerAtMove == 1:
            self.canonized = False
            return self.copy()
        else:
            temp = self.copy()
            temp.canonized = not self.canonized
            temp.internalArray = self.internalArray * self.playerAtMove
            # inverse
            temp.unusedPieces = [self.unusedPieces[1], self.unusedPieces[0]]
            temp.playerAtMove = 1
            # also revert for history as well
            return temp

    def getCrtPlayer(self) -> int:
        return self.playerAtMove

    # number of unused pieces for the player
    def getUnusedPlayerCount(self, player):
        return self.unusedPieces[(player + 1) // 2]

    def decUnusedPlayerCount(self, player):
        self.unusedPieces[(player + 1) // 2] -= 1

    # total number of pieces for the player
    def getPlayerCount(self, player):
        no_pieces_on_board = len([0 for pos in range(self.boardSize) if self.internalArray[pos] == player])
        return no_pieces_on_board + self.getUnusedPlayerCount(player)

    def getPosition(self, pos):
        return 0 if pos not in mcts2.Moara.VALID_POSITIONS else self.internalArray[
            mcts2.Moara.VALID_POSITIONS.index(pos)]

    def isInAMill(self, pos, player):
        # find all mills that contain the pos
        mill_list = self.getMills(pos)
        return list(filter(lambda x: self.getMillSum(x) == 3 * player, mill_list)) != []

    # get the mills that contain position
    def getMills(self, pos):
        return list(filter(lambda mill: list(filter(lambda p: p == pos, mill)) != [], self.MILLS))

    # get the sum of content of the mill
    def getMillSum(self, mill):
        return (self.internalArray[mill[0]] +
                self.internalArray[mill[1]] +
                self.internalArray[mill[2]])

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
        s = self.toShortString()
        # no more than 3 repetitions allowed:win if the opponent choose a state already existing
        if s in self.history and self.history[s] > 3:
            return 0.00000000001
        # no more than 50 moves without capture
        if self.noMovesWithoutCapture > 50:
            return 0.00000000001  # draw
        if self.getPlayerCount(player) < 3:
            return -1
        if self.getPlayerCount(-player) < 3:
            return 1
        player_valid_moves_list = self.getValidMoves(player)
        if player_valid_moves_list == []:
            return -1
        return 0

        # list of legal moves from the current position for the player

    # add reward for capture
    def getExtraReward(self):
        # return 0
        if self.noMovesWithoutCapture == 1 and self.noMoves > 1:
            return 0.5
        else:
            return 0.0

    def getValidMoves(self, player):
        s = self.toShortString()
        orig = -1
        dest = -1
        if player == -1:
            # revert the board.
            invariantBoard = self.copy()
            invariantBoard.internalArray = self.internalArray * player
            # inverse
            invariantBoard.unusedPieces = [self.unusedPieces[1], self.unusedPieces[0]]
            invariantBoard.playerAtMove = 1
            player = 1
        else:
            invariantBoard = self

        # memoization
        board_status = str(invariantBoard)
        if board_status in MoaraNew.ValidMovesFromState:
            return MoaraNew.ValidMovesFromState[board_status]

        result = []
        moves = []
        capture = []

        if invariantBoard.getUnusedPlayerCount(player) > 0:  # put
            # phase 1: can put anywhere where there is an empty place
            moves = [x for x in range(invariantBoard.boardSize) if invariantBoard.internalArray[x] == 0]
        else:
            if invariantBoard.getPlayerCount(player) > 3:  # move
                valid_moves = list(filter(lambda x: invariantBoard.internalArray[x[0]] == player and
                                                    invariantBoard.internalArray[x[1]] == 0,
                                          invariantBoard.VALID_MOVES))
                for v_m in valid_moves:
                    # transform into index in action moves
                    moves.append(invariantBoard.boardSize + v_m[0] * invariantBoard.boardSize + v_m[1])
            else:  # jump
                remaining = [x for x in range(invariantBoard.boardSize) if invariantBoard.internalArray[x] == player]
                free = [x for x in range(invariantBoard.boardSize) if invariantBoard.internalArray[x] == 0]
                for r in remaining:
                    for f in free:
                        moves.append(invariantBoard.boardSize + r * invariantBoard.boardSize + f)

        # for each possible move, determine if it forms a mill
        # if so then can capture any of the opponent pieces, that are not in a mill
        for move in moves:
            # extract destination:
            dest = move % invariantBoard.boardSize
            # there is also an origin to the move:
            if (move % invariantBoard.possibleMovesSize) > invariantBoard.boardSize:
                orig = (move % invariantBoard.possibleMovesSize) // invariantBoard.boardSize - 1;
            mills = invariantBoard.getMills(dest)  # any pos is in two mills
            # wouldBeInMill: bool = (invariantBoard.getMillSum(
            #     mills[0]) + invariantBoard.playerAtMove == 3 * invariantBoard.playerAtMove) or \
            #                       (invariantBoard.getMillSum(
            #                           mills[1]) + invariantBoard.playerAtMove == 3 * invariantBoard.playerAtMove)
            wouldBeInMill = False

            for mill in mills:
                copy_of_internal_array = np.array(invariantBoard.internalArray)
                copy_of_internal_array[dest] = invariantBoard.playerAtMove
                if orig != -1:
                    copy_of_internal_array[orig] = 0
                sum_mill = copy_of_internal_array[mill[0]] + \
                      copy_of_internal_array[mill[1]] + \
                      copy_of_internal_array[mill[2]]

                if sum_mill == 3 * invariantBoard.playerAtMove:
                    wouldBeInMill = True
                    break

            if wouldBeInMill:
                debug = 0
                if debug:
                    tmp = invariantBoard.copy()
                    tmp.internalArray[dest] = invariantBoard.playerAtMove
                    tmp.display()
                # find all opponents available to capture
                all_opponent_pieces = \
                    [x for x in range(invariantBoard.boardSize) if
                     invariantBoard.internalArray[x] == -invariantBoard.playerAtMove]
                # that is not an enemy mill
                available_opponent_pieces = list(
                    filter(lambda p: invariantBoard.isInAMill(p, -invariantBoard.playerAtMove) is False,
                           all_opponent_pieces))
                if len(available_opponent_pieces) == 0:
                    # if no available enemy piece to capture outside of aa mill
                    # retry with a piece from an enemy mill
                    # invariantBoard.display()
                    available_opponent_pieces = all_opponent_pieces
                # for each available opponent piece to be captured
                for x in available_opponent_pieces:
                    index = all_opponent_pieces.index(x)
                    result.append(move + (index + 1) * invariantBoard.possibleMovesSize)
            else:  # no mill, no capture
                result.append(move)
        MoaraNew.ValidMovesFromState[board_status] = result
        # self.SaveValidMoves()
        return result

    def getNextState(self, action):
        # state will be modified in the copied object
        newGameState = self.copy()
        newGameState.playerAtMove = -self.playerAtMove
        # newGameState.canonized = False
        if self.getUnusedPlayerCount(self.playerAtMove) > 0:  # put
            piece_to_put = action % self.possibleMovesSize
            newGameState.internalArray[piece_to_put] = self.playerAtMove
            newGameState.decUnusedPlayerCount(self.playerAtMove)
            # also capture
            if action > self.possibleMovesSize:
                # newGameState.display()
                opponent_piece_index = action // self.possibleMovesSize - 1
                all_opponent_pieces = \
                    [x for x in range(self.boardSize) if newGameState.internalArray[x] == -self.playerAtMove]
                opponent_piece = all_opponent_pieces[opponent_piece_index]
                assert (newGameState.internalArray[opponent_piece] == -self.playerAtMove)
                newGameState.internalArray[opponent_piece] = 0
                newGameState.noMovesWithoutCapture = 0  # reset it

        else:
            piece_to_move = action % self.possibleMovesSize
            orig = piece_to_move // self.boardSize - 1
            assert (newGameState.internalArray[orig] == self.playerAtMove)
            dest = piece_to_move % self.boardSize
            assert (newGameState.internalArray[dest] == 0)
            newGameState.internalArray[dest] = self.playerAtMove
            newGameState.internalArray[orig] = 0
            # also capture
            if action > self.possibleMovesSize:
                # newGameState.display()
                opponent_piece_index = action // self.possibleMovesSize - 1
                all_opponent_pieces = \
                    [x for x in range(self.boardSize) if newGameState.internalArray[x] == -self.playerAtMove]
                opponent_piece = all_opponent_pieces[opponent_piece_index]
                assert (newGameState.internalArray[opponent_piece] == -self.playerAtMove)
                newGameState.internalArray[opponent_piece] = 0
                newGameState.noMovesWithoutCapture = 0  # reset it

        newGameState.updateHistory()
        # newGameState.display()
        return newGameState

    def updateHistory(self):
        s = self.toShortString()
        if self.canonized:
            # transform s
            l = list(s)
            l = ['x' if x == 'o' else 'o' if x == 'x' else '_' for x in l]
            s = ''.join(l)
            # s = self.toShortString()
        if s not in self.history:
            self.history[s] = 1
        else:
            self.history[s] += 1
        self.moves.append(s)
        self.noMoves += 1
        self.noMovesWithoutCapture += 1

    def display(self):
        n = 7
        print("")
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print(
            f"X={self.getPlayerCount(1)}[{self.getUnusedPlayerCount(1)}] Y={self.getPlayerCount(-1)}[{self.getUnusedPlayerCount(-1)}]. Move #{self.noMoves}",
            end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = self.getPosition((y, x)) * 1  # get the piece to print
                if piece == 1:
                    print("X ", end="")
                elif piece == -1:
                    print("O ", end="")
                else:
                    if (y, x) in mcts2.Moara.VALID_POSITIONS:
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
                        if x == n:
                            print(" ", end="")
                        else:
                            print("  ", end="")

            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")

    def getInternalRepresentation(self):
        result = []
        # planes = (8 * (1 +  # board for player 1
        #                     1 +  # board for player 2
        #                     1 +  # rotated board for player 1
        #                     1 +  # rotated board for player 2
        #                     1 +  # unused pieces for player 1
        #                     1) +  # unused pieces for player 2
        #                1 +  # no repetitions for current board, uniform value over array
        #                1 +  # total moves, uniform value over array
        #                1)  # moves without capture, uniform value over array
        #

        # propagate back in temporal array
        for i in range(self.time_len - 1):  # i = 0, is the oldest
            for j in range(self.feature_len):
                self.timePlanes[i][j] = self.timePlanes[i + 1][j]
        # normal board for player
        self.timePlanes[self.time_len - 1][0] = [1 if x == self.playerAtMove else 0 for x in self.internalArray]
        # normal board for opponent
        self.timePlanes[self.time_len - 1][1] = [1 if x == -self.playerAtMove else 0 for x in self.internalArray]
        arr2 = np.array(self.MILLS).reshape(48)
        # rotate board
        rot = [self.internalArray[arr2[x + self.boardSize]] for x in range(self.boardSize)]
        # rotated board for player
        self.timePlanes[self.time_len - 1][2] = [1 if x == self.playerAtMove else 0 for x in rot]
        # rotated board for opponent
        self.timePlanes[self.time_len - 1][3] = [1 if x == -self.playerAtMove else 0 for x in rot]
        # unused pieces for player
        self.timePlanes[self.time_len - 1][4] = [self.getUnusedPlayerCount(self.playerAtMove) for _ in
                                                 range(self.boardSize)]
        # unused pieces for opponent
        self.timePlanes[self.time_len - 1][5] = [self.getUnusedPlayerCount(-self.playerAtMove) for _ in
                                                 range(self.boardSize)]
        # append time data to result
        for i in range(self.time_len):
            for j in range(self.feature_len):
                result.append(self.timePlanes[i][j])
        # repetition
        s = self.toShortString()
        rep = self.history[s] if s in self.history else 0
        result.append([rep for _ in range(self.boardSize)])
        # total moves
        result.append([self.noMoves for _ in range(self.boardSize)])
        # total moves without capture
        result.append([self.noMovesWithoutCapture for _ in range(self.boardSize)])
        res2 = np.array(result).reshape(51, self.board_X, self.board_Y)
        return res2


print("mcts 3")
moaraGame: MoaraNew = MoaraNew()
moaraGame.LoadValidMoves()
n = NeuralNetNew(moaraGame, mcts2.moara.args)
n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)

mcts = mcts2.MCTS(n)
mcts2.learn(moaraGame, mcts, n)
