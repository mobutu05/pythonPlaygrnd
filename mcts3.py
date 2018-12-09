import copy
import functools
import os
import pickle
from collections import deque

import numpy as np
from keras import Input, Model
from keras.layers import BatchNormalization, Reshape, Activation, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from numpy.random.mtrand import shuffle

import mcts2
import moara
import matplotlib.pyplot as plt
from mcts2 import IGame

class Player:
    def __init__(self, name, method):
        self.name = name
        self.method = method
        self.score = 0

    def __repr__(self):
        return self.name

class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1: Player, player2: Player, game: IGame, args, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1:Player = player1
        self.player2:Player = player2

        self.game: IGame = game
        self.display = display
        self.args = args

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        it = 0
        trainExamples = []
        self.game.reset()
        r = 0
        NRs = {}
        while r == 0:
            it += 1
            action = players[self.game.getCrtPlayer() + 1].method(self.game)
            p = [1 if x == action else 0 for x in range(self.game.getActionSize())]
            trainExamples.append([self.game.getInternalRepresentation(), self.game.getCrtPlayer(), p])
            self.game = self.game.getNextState(action)
            r = self.game.getGameEnded()
            if it > 1000:
                r = 0.001

            if verbose:
                self.game.display()
                print(f"Turn {it:03d} {str(self.game)} Player { players[-self.game.getCrtPlayer() + 1].name}")
            else:
                print('.',end="")

        if verbose:
            self.game.display()
            print(f"Game over: Turn {it}. Result {self.player1.name}-{self.player2.name} is {r}")
        self.iterationTrainExamples = [(x[0], x[2], r * ((-1) ** (x[1] != self.game.getCrtPlayer()))) for x in
                                       trainExamples]
        return r

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        self.trainExamplesHistory = []

        eps = 0
        maxeps = int(num)

        # num = int(num / 2)
        draws = 0
        inverse = False
        for i in range(num):
            self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                self.player1.score += 1
            elif gameResult == -1:
                self.player2.score += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            self.trainExamplesHistory.append(self.iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                # print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                #       " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            print('-')
            print(f"Round {i}: {gameResult}; {self.player1.name}:{self.player1.score}  {self.player2.name}:{self.player2.score} - {draws}")

            self.player1, self.player2 = self.player2, self.player1

            if self.player1.score > num / 2 or self.player2.score > num / 2:
                break
        # for i in range(num):
        #     gameResult = self.playGame(verbose=verbose)
        #     if gameResult == -1:
        #         oneWon += 1
        #     elif gameResult == 1:
        #         twoWon += 1
        #     else:
        #         draws += 1
        #     # bookkeeping + plot progress
        #     print(f"Round {i}: {gameResult};  {oneWon} - {twoWon} - {draws}")
        #
        #
        #     self.trainExamplesHistory.append(self.iterationTrainExamples)
        #     if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
        #         # print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
        #         #       " => remove the oldest trainExamples")
        #         self.trainExamplesHistory.pop(0)

        return draws


class RandomPlayer():
    def __init__(self):
        pass

    def play(self, game):
        valids = game.getValidMoves(game.getCrtPlayer())
        shuffle(valids)
        return valids[0]


class HumanPlayer():
    def __init__(self):
        pass

    def play(self, game):
        # display(board)
        valid = game.getValidMoves(game.getCrtPlayer())

        while True:
            input_string = input()
            try:
                input_array = [int(x) for x in input_string.split(' ')]
            except:
                input_array = []
            # put
            if len(input_array) == 2:
                a, b = input_array
                try:
                    move = moara.Game.validPositions.index((a, b))
                except:
                    move = -1
            elif len(input_array) == 4:
                a, b, c, d = input_array
                try:
                    pos1 = moara.Game.validPositions.index((a, b))
                    pos2 = moara.Game.validPositions.index((c, d))
                    # 1 move/jump
                    move = game.boardSize + pos1 * game.boardSize + pos2
                    if move not in valid:
                        # 2 capture
                        # find all opponents
                        all_opponent_pieces = [x for x in range(game.boardSize) if
                                               game.internalArray[x] == -game.playerAtMove]
                        index = all_opponent_pieces.index(pos2)
                        move = pos1 + (index + 1) * game.possibleMovesSize
                except:
                    move = -1
            elif len(input_array) == 6:
                a, b, c, d, e, f = input_array
                try:
                    # move/jump and capture
                    pos1 = moara.Game.validPositions.index((a, b))
                    pos2 = moara.Game.validPositions.index((c, d))
                    pos3 = moara.Game.validPositions.index((e, f))
                    move = game.boardSize + pos1 * game.boardSize + pos2
                    all_opponent_pieces = [x for x in range(game.boardSize) if
                                           game.internalArray[x] == -game.playerAtMove]
                    index = all_opponent_pieces.index(pos3)
                    move = move + (index + 1) * game.possibleMovesSize
                except:
                    move = -1
            else:
                move = 90  # invalid

            if move in valid:
                break
            else:
                print('Invalid')

        return move


class MoaraNew:
    pass


class NeuralNetNew(mcts2.INeuralNet):
    def __init__(self, game: MoaraNew, args):
        self.args = args
        self.action_size = game.getActionSize()
        self.board_x = game.board_X  # lines
        self.board_y = game.board_Y  # columns
        self.planes = (game.time_len * game.feature_len +
                       1 +  # no repetitions for current board, uniform value over array
                       1 +  # total moves, uniform value over array
                       1)  # moves without capture, uniform value over array

        self.input_boards = Input(shape=(self.planes, self.board_x, self.board_y))  # s: batch_size x board_x x board_y

        x_image = BatchNormalization(axis=3)(
            Reshape((self.board_x, self.board_y, self.planes))(
                self.input_boards))  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
            x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
            h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
            h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(
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


class MoaraNew(mcts2.IGame):
    # transition from one position to another
    VALID_MOVES = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.VALID_MOVES]
    MILLS = [list(map(lambda x: mcts2.Moara.VALID_POSITIONS.index(x), y)) for y in mcts2.Moara.MILLS]
    # list of possible moves from each board configuration
    # this is independent of a particular game, so can be serialized
    ValidMovesFromState = {}

    def __init__(self, ):
        self.board_X = 8
        self.board_Y = 3
        self.boardSize = self.board_X * self.board_Y
        self.unusedPieces = [9, 9]  # for opponent, player
        self.playerAtMove = 1
        self.internalArray = np.array([0 for _ in range(self.boardSize)])

        # number of moves since last capture
        self.noMovesWithoutCapture = 0
        # number of total moves
        self.noMoves = 0
        self.time_len = 1  # keep current state and previous 0
        self.feature_len = 7  # number of planes describing current board
        self.possibleMovesSize = (self.boardSize +  # put pieces (while unused pieces exist)
                                  self.boardSize * self.boardSize)  # move/jump pieces

    def SaveData(self):
        pass

    def reset(self):
        self.__init__()

    # string representation of the current position in the game
    def __repr__(self):
        board = ''.join(['x' if x == 1 else 'o' if x == -1 else '_' for x in self.internalArray])
        player = 'x' if self.playerAtMove == 1 else 'o'
        return f"{board}{self.getUnusedPlayerCount(1)}{self.getUnusedPlayerCount(-1)}{player}"

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
            temp.canonized = True
            temp.internalArray = np.array(
                [self.internalArray[0] * self.playerAtMove, self.internalArray[2], self.internalArray[1],
                 self.internalArray[3] * self.playerAtMove])
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
        s = str(self)
        # no more than 50 moves without capture
        if self.noMovesWithoutCapture > 50:
            return 0.00000000001  # draw
        if self.getPlayerCount(self.playerAtMove) < 3:
            return -1
        if self.getPlayerCount(-self.playerAtMove) < 3:
            return 1
        player_valid_moves_list = self.getValidMoves(self.playerAtMove)
        if player_valid_moves_list == []:
            return -1
        return 0

        # list of legal moves from the current position for the player

    # add reward for capture
    def getExtraReward(self):
        return 0
        # if self.noMovesWithoutCapture == 1 and self.noMoves > 1:
        #     return 1
        # else:
        #     return 0.0

    def getValidMoves(self, player):
        orig = -1
        dest = -1

        # memoization
        board_status = str(self)
        if board_status in MoaraNew.ValidMovesFromState:
            # if s not in invariantBoard.history or invariantBoard.history[s] < 2:
            return MoaraNew.ValidMovesFromState[board_status]
            # else:

            # need to simulate
            # if the position has already been repeated than make sure that no subsequent move
            # simulate each move, so that it wouldn't get into a invalid condition
            # 3 state repetition or 50 moves without capture
            # possibleMoves = []
            # for move in MoaraNew.ValidMovesFromState[board_status]:
            #     # if (s, move) not in self.memo:
            #     if (board_status, move) not in MoaraNew.ValidMovesFromState:
            #         newState = self.getNextState(move)
            #         s = str(newState)
            #         MoaraNew.ValidMovesFromState[(board_status, move)] = s
            #     else:
            #         s = MoaraNew.ValidMovesFromState[(board_status, move)]
            #     if (s not in self.history or self.history[s] < 3) and self.noMovesWithoutCapture < 50:
            #         possibleMoves.append(move)
            #
            # return possibleMoves

        result = []
        moves = []
        capture = []

        if self.getUnusedPlayerCount(player) > 0:  # put
            # phase 1: can put anywhere where there is an empty place
            moves = [x for x in range(self.boardSize) if self.internalArray[x] == 0]
        else:
            if self.getPlayerCount(player) > 3:  # move
                valid_moves = list(filter(lambda x: self.internalArray[x[0]] == player and
                                                    self.internalArray[x[1]] == 0,
                                          self.VALID_MOVES))
                for v_m in valid_moves:
                    # transform into index in action moves
                    moves.append(self.boardSize + v_m[0] * self.boardSize + v_m[1])
            else:  # jump
                remaining = [x for x in range(self.boardSize) if self.internalArray[x] == player]
                free = [x for x in range(self.boardSize) if self.internalArray[x] == 0]
                for r in remaining:
                    for f in free:
                        moves.append(self.boardSize + r * self.boardSize + f)

        # for each possible move, determine if it forms a mill
        # if so then can capture any of the opponent pieces, that are not in a mill
        for move in moves:
            # extract destination:
            dest = move % self.boardSize
            # there is also an origin to the move:
            if (move % self.possibleMovesSize) > self.boardSize:
                orig = (move % self.possibleMovesSize) // self.boardSize - 1;
            mills = self.getMills(dest)  # any pos is in two mills
            # wouldBeInMill: bool = (invariantBoard.getMillSum(
            #     mills[0]) + invariantBoard.playerAtMove == 3 * invariantBoard.playerAtMove) or \
            #                       (invariantBoard.getMillSum(
            #                           mills[1]) + invariantBoard.playerAtMove == 3 * invariantBoard.playerAtMove)
            wouldBeInMill = False

            for mill in mills:
                copy_of_internal_array = np.array(self.internalArray)
                copy_of_internal_array[dest] = self.playerAtMove
                if orig != -1:
                    copy_of_internal_array[orig] = 0
                sum_mill = copy_of_internal_array[mill[0]] + \
                           copy_of_internal_array[mill[1]] + \
                           copy_of_internal_array[mill[2]]

                if sum_mill == 3 * self.playerAtMove:
                    wouldBeInMill = True
                    break

            if wouldBeInMill:
                debug = 0
                if debug:
                    tmp = self.copy()
                    tmp.internalArray[dest] = self.playerAtMove
                    tmp.display()
                # find all opponents available to capture
                all_opponent_pieces = \
                    [x for x in range(self.boardSize) if
                     self.internalArray[x] == -self.playerAtMove]
                # that is not an enemy mill
                available_opponent_pieces = list(
                    filter(lambda p: self.isInAMill(p, -self.playerAtMove) is False,
                           all_opponent_pieces))
                if len(available_opponent_pieces) == 0:
                    # if no available enemy piece to capture outside of aa mill
                    # retry with a piece from an enemy mill
                    # invariantBoard.display()
                    available_opponent_pieces = all_opponent_pieces
                # for each available opponent piece to be captured
                for x in available_opponent_pieces:
                    index = all_opponent_pieces.index(x)
                    result.append(move + (index + 1) * self.possibleMovesSize)
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
            if newGameState.internalArray[orig] != self.playerAtMove:
                assert (newGameState.internalArray[orig] == self.playerAtMove)
            dest = piece_to_move % self.boardSize
            if newGameState.internalArray[dest] != 0:
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
        s = str(self)
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
        planes = (8 * (1 +  # board for player 1
                       1 +  # board for player 2
                       1 +  # rotated board for player 1
                       1 +  # rotated board for player 2
                       1 +  # player color (+1 - white(first), -1 - black)
                       1 +  # unused pieces for player 1
                       1) +  # unused pieces for player 2
                  1 +  # no of repetitions for current board, uniform value over array
                  1 +  # total moves, uniform value over array
                  1)  # moves without capture, uniform value over arrays

        # propagate back in temporal array
        for i in range(self.time_len):  # i = 0, is the oldest
            # use historical moves, extract from history, if enough moves where made
            if len(self.moves) <= i:
                for j in range(self.feature_len):
                    result.append([0] * self.boardSize)
            else:
                l = list(self.moves[len(self.moves) - 1 - i])
                board = [1 if l[x] == 'x' else -1 if l[x] == '0' else 0 for x in range(self.boardSize)]
                unusedPlayer1 = int(l[self.boardSize])
                unusedPlayer2 = int(l[self.boardSize + 1])
                crtPlayer = 1 if l[self.boardSize + 2] == 'x' else -1

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
                # color
                result.append([crtPlayer] * self.boardSize)

                # unused pieces for player 1
                result.append([unusedPlayer1] * self.boardSize)
                # unused pieces for player 2
                result.append([unusedPlayer2] * self.boardSize)

        # repetition
        s = str(self)
        rep = 0 #self.history[s] if s in self.history else 0
        result.append([rep] * self.boardSize)
        # total moves
        result.append([self.noMoves] * self.boardSize)
        # total moves without capture
        result.append([self.noMovesWithoutCapture] * self.boardSize)
        res2 = np.array(result).reshape(planes, self.board_X, self.board_Y)
        return res2


def doArena(n: mcts2.INeuralNet, mcts: mcts2.MCTS, doTrain=True):
    # otherPlayer = lambda x: HumanPlayer().play(x)
    otherPlayer = Player("random", lambda x: RandomPlayer().play(x))
    neuralPlayer = Player("neural", lambda x: np.argmax(mcts.getActionProbabilities(x, 0)))
    a = Arena(neuralPlayer, otherPlayer, moaraGame, moara.args, mcts)

    result = a.playGames(10, verbose=False)
    if doTrain:
        # train the network based on the arena games
        trainExamples = []
        for e in a.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)
        if trainExamples != []:
            n.train(trainExamples)

            # test against the previous
            # if i % 5 == 0:
            #     # self.PitAgainst('no36.neural.data-ITER-390')
            #     PitAgainst(moara.filename - 1)
            n.save_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)


print("mcts 3")
moaraGame: MoaraNew = MoaraNew()
# moaraGame.LoadValidMoves()
n = NeuralNetNew(moaraGame, mcts2.moara.args)
n.load_checkpoint(folder=moara.args.checkpoint, filename_no=moara.args.filename)

mcts = mcts2.MCTS(n)
mcts2.learn(moaraGame, mcts, n, doArena)

# doArena(n, mcts)
