import functools
import math
import os
from collections import deque
from random import shuffle

import numpy as np
from keras import Input, Model
from keras.layers import Reshape, Activation, Conv2D, BatchNormalization, Flatten, Dense, Dropout, LSTM, ConvLSTM2D, GRU
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import copy

EPS = 1e-8


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, args, display=None):
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
        self.player1 = player1
        self.player2 = player2
        self.game = game
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
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        trainExamples = []
        r = 0
        NRs = {}
        while r == 0:
            it += 1

            canonicalBoard = board.getCanonicalForm(curPlayer)
            s = str(canonicalBoard)

            action = players[curPlayer + 1](canonicalBoard)

            valids = self.game.getValidMoves(canonicalBoard, 1)
            if verbose:
                if valids[action] == 0:
                    print(action)
                    print(valids)
                    board.display(1)
                    print(str(board))
                    assert valids[action] > 0
            p = [1 if x == action else 0 for x in range(self.game.getActionSize())]
            trainExamples.append([canonicalBoard.internalArray, curPlayer, p])
            validActions = list(filter(lambda x: valids[x] != 0, [i for i in range(self.game.getActionSize())]))
            while True:
                new_board, new_curPlayer = self.game.getNextState(board, curPlayer, action)

                s = str(new_board.getCanonicalForm(new_curPlayer))
                if s not in NRs:
                    NRs[s] = 1
                else:
                    NRs[s] += 1

                if NRs[s] < 2 and it < 1000:
                    break
                else:
                    if verbose:
                        print(f"Action {action} lead to duplicate positions")
                        print(validActions)
                    if len(validActions) > 1:
                        validActions.remove(action)
                    action = np.random.choice(validActions)
                    if verbose:
                        print(f"Randomly select {action}")
                    if len(validActions) <= 1:
                        break
                    # if s not in NRs or NRs[s] < 2:
                    #     pass
                    # else:
                    #     print(f"Action {action} also lead to duplicate position. Retry")

            board = new_board
            curPlayer = new_curPlayer
            r = self.game.getGameEnded(board, 1)
            if it > 1000:
                r = 0.001
            if verbose:
                print(f"Turn {it:03d} {str(board)} Player {curPlayer}")
                # board.display(1)
        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(r))
            board.display(curPlayer)
        self.iterationTrainExamples = [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
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
        oneWon = 0
        twoWon = 0
        draws = 0
        inverse = False
        for i in range(num):
            self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                if inverse:
                    twoWon += 1
                else:
                    oneWon += 1
            elif gameResult == -1:
                if not inverse:
                    twoWon += 1
                else:
                    oneWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            self.trainExamplesHistory.append(self.iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                # print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                #       " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            print(f"Round {i}: {gameResult};  {oneWon} - {twoWon} - {draws}")

            self.player1, self.player2 = self.player2, self.player1
            inverse = not inverse
            # oneWon, twoWon = twoWon, oneWon
            if oneWon > num / 2 or twoWon > num / 2:
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

        return oneWon, twoWon, draws


class Board():
    def __init__(self, copy_=None):
        if copy_ is None:
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
            arr2 = np.array([[0. for y in range(7)] for x in range(7)])
            arr2[3][3] = 9.0  # player/opponent
            # third plane - number of pieces for player 2
            arr3 = np.array([[0. for y in range(7)] for x in range(7)])
            arr3[3][3] = 9.0  # player/opponent
            # fourth plane - flag is current player must capture
            arr4 = np.array([[0. for y in range(7)] for x in range(7)])
            arr4[3][3] = 0.0
            self.internalArray = np.array([arr1, arr2, arr3, arr4])
        else:
            self.internalArray = np.copy(copy_)

    def getCanonicalForm(self, player):
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
        if player == 1:
            return Board(self.internalArray)
        else:
            b = Board(np.array([self.internalArray[0] * player, self.internalArray[2], self.internalArray[1],
                                self.internalArray[3] * player]))
            return b

    def setPosition(self, pos, value):
        (x, y) = pos
        self.internalArray[0][y][x] = value

    def getPosition(self, pos):
        (x, y) = pos
        return self.internalArray[0][y][x]

    def getPlayerCount(self, player):
        if player == 1:
            return int(round(self.internalArray[1][3][3]))
        else:
            return int(round(self.internalArray[2][3][3]))

    def setPlayerCount(self, player, count):
        if player == 1:
            self.internalArray[1][3][3] = count
        else:
            self.internalArray[2][3][3] = count

    def getOpponentCount(self, player):
        if player == 1:
            return int(round(self.internalArray[2][3][3]))
        else:
            return int(round(self.internalArray[1][3][3]))

    def setOpponentCount(self, player, count):
        if player == 1:
            self.internalArray[2][3][3] = count
        else:
            self.internalArray[1][3][3] = count

    def getBoardStatus(self):
        return int(round(self.internalArray[3][3][3]))

    def setBoardStatus(self, status):
        self.internalArray[3][3][3] = status

    def display(self, current_player, invariant=1):
        n = self.internalArray.shape[1]
        print("")
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = self.getPosition((x, y)) * current_player * invariant  # get the piece to print
                if piece == 1:
                    print("X ", end="")
                elif piece == -1:
                    print("O ", end="")
                else:
                    if (y, x) in Game.validPositions:
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

    def getShortString(self):
        hh = ''
        for (x, y) in Game.validPositions:
            if self.getPosition((x, y)) == 1:
                hh += "x"
            elif self.getPosition((x, y)) == -1:
                hh += "o"
            else:
                hh += "_"
        return hh

    def __repr__(self):
        hh = ''
        for (x, y) in Game.validPositions:
            if self.getPosition((x, y)) == 1:
                hh += "x"
            elif self.getPosition((x, y)) == -1:
                hh += "o"
            else:
                hh += "_"
        hh = hh + " "
        hh = hh + str(self.getPlayerCount(1))  # player
        hh = hh + " "
        hh = hh + str(self.getOpponentCount(1))  # opponent
        hh = hh + " "
        hh = hh + str(self.getBoardStatus())  # capture
        return hh


class Board2:
    def __init__(self, copy_=None):
        if copy_ is None:
            self.internalArray = np.array([
                # first plane - white pieces - time n
                np.array([[0. for y in range(7)] for x in range(7)]),
                # first plane - white pieces - time n-1
                np.array([[0. for y in range(7)] for x in range(7)])
                # # first plane - white pieces - time n-1
                # arr2 = np.array([[0. for y in range(7)] for x in range(7)])
                #
                #
                #
                # arr2 = np.array([[0. for y in range(7)] for x in range(7)])
                # arr2[3][3] = 9.0  # player/opponent
                # # third plane - number of pieces for player 2
                # arr3 = np.array([[0. for y in range(7)] for x in range(7)])
                # arr3[3][3] = 9.0  # player/opponent
                # # fourth plane - flag is current player must capture
                # arr4 = np.array([[0. for y in range(7)] for x in range(7)])
                # arr4[3][3] = 0.0
            ])
        else:
            self.internalArray = np.copy(copy_)

    def getCanonicalForm(self, player):
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
        if player == 1:
            return Board(self.internalArray)
        else:
            b = Board(np.array([self.internalArray[0] * player, self.internalArray[2], self.internalArray[1],
                                self.internalArray[3] * player]))
            return b

    def setPosition(self, pos, value):
        (x, y) = pos
        self.internalArray[0][y][x] = value

    def getPosition(self, pos):
        (x, y) = pos
        return self.internalArray[0][y][x]

    def getPlayerCount(self, player):
        if player == 1:
            return int(round(self.internalArray[1][3][3]))
        else:
            return int(round(self.internalArray[2][3][3]))

    def setPlayerCount(self, player, count):
        if player == 1:
            self.internalArray[1][3][3] = count
        else:
            self.internalArray[2][3][3] = count

    def getOpponentCount(self, player):
        if player == 1:
            return int(round(self.internalArray[2][3][3]))
        else:
            return int(round(self.internalArray[1][3][3]))

    def setOpponentCount(self, player, count):
        if player == 1:
            self.internalArray[2][3][3] = count
        else:
            self.internalArray[1][3][3] = count

    def getBoardStatus(self):
        return int(round(self.internalArray[3][3][3]))

    def setBoardStatus(self, status):
        self.internalArray[3][3][3] = status

    def display(self, current_player, invariant=1):
        n = self.internalArray.shape[1]
        print("")
        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = self.getPosition((x, y)) * current_player * invariant  # get the piece to print
                if piece == 1:
                    print("X ", end="")
                elif piece == -1:
                    print("O ", end="")
                else:
                    if (y, x) in Game.validPositions:
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

    def getShortString(self):
        hh = ''
        for (x, y) in Game.validPositions:
            if self.getPosition((x, y)) == 1:
                hh += "x"
            elif self.getPosition((x, y)) == -1:
                hh += "o"
            else:
                hh += "_"
        return hh

    def __repr__(self):
        hh = ''
        for (x, y) in Game.validPositions:
            if self.getPosition((x, y)) == 1:
                hh += "x"
            elif self.getPosition((x, y)) == -1:
                hh += "o"
            else:
                hh += "_"
        hh = hh + " "
        hh = hh + str(self.getPlayerCount(1))  # player
        hh = hh + " "
        hh = hh + str(self.getOpponentCount(1))  # opponent
        hh = hh + " "
        hh = hh + str(self.getBoardStatus())  # capture
        return hh


class IGame:
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, board, player, action):
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

    def getValidMoves(self, board, player):
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

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def getLegalMoves(self, board, player):
        pass


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    # valid position where pieces reside
    validPositions = [(0, 0), (0, 3), (0, 6),
                      (1, 1), (1, 3), (1, 5),
                      (2, 2), (2, 3), (2, 4),
                      (3, 0), (3, 1), (3, 2),
                      (3, 4), (3, 5), (3, 6),
                      (4, 2), (4, 3), (4, 4),
                      (5, 1), (5, 3), (5, 5),
                      (6, 0), (6, 3), (6, 6)]
    # transition from one position to another
    validActions = [
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

    mills = [
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

    def __init__(self):
        self.n = 7

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return Board()

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
                len(self.validActions) +  # move piece
                1)  # pass

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # if player takes action on board, return next (board,player)

        # if player must make a move or capture, i.e. board status != 0,
        # the opponent takes a dummy move, nothing changes
        if action == 24 + len(Game.validActions):  # pass
            return board, -player

        # action must be a valid move
        board = Board(board.internalArray)
        pos = (3, 3)
        pieces_on_board = len([0 for pos in Game.validPositions if board.getPosition(pos) == player])
        player_no = board.getPlayerCount(player)

        # pre-selection for capture
        if abs(board.getBoardStatus()) == 100:  # capture
            # could not capture, but can move
            if action >= 24:
                board.setBoardStatus(0)
            # could not capture, but can jump
            else:
                move = Game.validPositions[action]
                if board.getPosition(move) != -player:
                    # board.display(player)
                    board.setBoardStatus(0)

        # phase 1
        if board.getBoardStatus() == 0:  # select/put piece

            if player_no > pieces_on_board:  # put
                pos = Game.validPositions[action]
                board.setPosition(pos, player)
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
        if self.isInAMill(board, pos, player):
            board.setBoardStatus(100 * player)  # flag that a capture can be made
        return board, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # make it a 0 and 1 array for each legal move
        moves = self.getLegalMoves(board, player)
        # return list(map(lambda x: 1 if x in moves else 0, validActions))
        return [1 if x in moves else 0 for x in range(self.getActionSize())]

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player win if:
        #   - opponent has less than 3 pieces
        #   - opponent has no valid moves
        # draw if:
        #   - no valid moves for any player
        #   - future:
        #       - 50 moves with no capture
        #       - position replay 3 times
        player_valid = self.getLegalMoves(board, player)
        opponent_valid = self.getLegalMoves(board, -player)
        if player_valid == [] and opponent_valid == []:
            return 0.001  # draw
        if board.getOpponentCount(player) < 3 or opponent_valid == []:
            return 1
        if board.getPlayerCount(player) < 3 or player_valid == []:
            return -1
        return 0

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # mirror, rotational
        # pi_board = np.reshape(pi[:-1], (self.n, self.n))
        # l = []
        #
        # for i in range(1, 5):
        #     for j in [True, False]:
        #         newB = np.rot90(board, i)
        #         newPi = np.rot90(pi_board, i)
        #         if j:
        #             newB = np.fliplr(newB)
        #             newPi = np.fliplr(newPi)
        #         l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        # return l
        return [(board, pi)]

    def isMill(self, board, mill, player):
        count = functools.reduce(lambda acc, i: acc + (1 if board.getPosition(mill[i]) == player else 0),
                                 range(3), 0)

        if count == 3:
            return 1
        else:
            return 0

    # is the piece in a mill of a player?
    def isInAMill(self, board, pos, player):
        # find all mills that contain the pos
        mill_list = list(filter(lambda mill: list(filter(lambda p: p == pos, mill)) != [], Game.mills))
        return list(filter(lambda x: self.isMill(board, x, player) == 1, mill_list)) != []

    def getLegalMoves(self, board, player):
        # board = Board(board.internalArray)
        boardStatus = board.getBoardStatus()
        if boardStatus < 0:
            xxx = 0

        # opposite player passes
        if boardStatus != 0 and np.sign(boardStatus) != np.sign(player):
            return [24 + len(Game.validActions)]  # pass

        # only if the last move results in a mill
        # if the player has a mill, it must remove an opponent's piece, that is not a mill either
        # mill_no = functools.reduce(lambda acc, mill: acc + isMill(board, mill, player), mills, 0)
        # need to select an opponent piece
        if boardStatus == 100 * player:
            available_opponent_pieces = list(
                filter(lambda x: board.getPosition(x) == -player, Game.validPositions))
            result = list(filter(lambda p: self.isInAMill(board, p, -player) is False, available_opponent_pieces))
            result = [Game.validPositions.index(x) for x in result]
            if len(result) > 0:
                return result  # else choose another move
            else:
                # print("can't capture")
                # board.setBoardStatus(0)
                boardStatus = 0
        # move piece, select destination, phase 2 or 3

        # pieces_on_board = functools.reduce(lambda acc, pos: acc + (1 if self.getPosition(board, pos) == player else 0),
        #                                    self.validPositions, 0)
        pieces_on_board = len([0 for pos in self.validPositions if board.getPosition(pos) == player])
        result = []

        player_no = board.getPlayerCount(player)

        if boardStatus != 0:
            # select those actions that originate in the stored position
            boardStatus = boardStatus * player
            if player_no > 3:  # move
                # (x, y) = Game.validPositions[boardStatus - 1]
                # result = list(filter(lambda a: a[0] == (x, y), Game.validActions))
                # result = [x[1] for x in result if board.getPosition(x[1]) == 0]
                # result = set(result)
                # result = [Game.validPositions.index(x) for x in result]
                result = list(filter(lambda x: board.getPosition(x[0]) == player and
                                               board.getPosition(x[1]) == 0,
                                     Game.validActions))
                result = [24 + Game.validPositions.index(x) for x in result]
            if player_no == 3:
                # any empty place
                result = list(filter(lambda x: board.getPosition(x) == 0, Game.validPositions))
                result = [Game.validPositions.index(x) for x in result]
            return result
        if player_no > pieces_on_board:  # there are still pieces to put on board
            # phase 1: can put anywhere where there is an empty place
            result = list(filter(lambda x: board.getPosition(x) == 0, Game.validPositions))
            # result = [3 * 24 + self.validPositions.index(x) for x in result]
            result = [Game.validPositions.index(x) for x in result]
        elif player_no > 3:
            # phase 2: need to move a piece in a valid position
            # select all transitions that have a player piece in the first position and an empty one in the second position
            result = list(filter(lambda x: board.getPosition(x[0]) == player and
                                           board.getPosition(x[1]) == 0, Game.validActions))
            # result = [x[0] for x in result]
            # result = set(result)
            result = [24 + Game.validActions.index(x) for x in result]
        elif player_no == 3:
            # phase 3: when only 3 pieces left can move anywhere empty
            result = list(filter(lambda x: board.getPosition(x) == player, Game.validPositions))
            result = [Game.validPositions.index(x) for x in result]
        return result

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.deep = 0
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.NRs = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard: Board, temperature: object = 1) -> int:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        for i in range(self.args.numMCTSSimulations):
            # print(" ")
            # print(" ")
            # print(".", end="")
            # print(".")

            # number of times each state occurs during this tree search
            NRs = {}  # clean counter for each move
            v = self.search(canonicalBoard, NRs)
            self.deep = 0

        # print ("")
        s = str(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        return counts

    def getBestAction(self, s, valid_actions):
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

        for a in valid_actions:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + \
                    self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a
        a = best_act
        return a

    def search(self, canonicalBoard, history):
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
        # self.game.display(canonicalBoard, 1)
        # print (self.deep)
        s = str(canonicalBoard)
        local_history = history.copy()
        if s not in local_history:
            local_history[s] = 1
        else:
            local_history[s] += 1
        if local_history[s] > 2:
            # print("draw")
            return 0.001  # draw through repetition

        self.deep += 1
        if self.deep > 1000:
            return 0
        # if s not in self.NRs:
        #     self.NRs[s] = 0
        # else:
        #     self.NRs[s] += 1
        # if self.NRs[s] > 3:
        #     return 0
        # print(s)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps or local_history[s] > 3:
            # leaf node

            ps, v = self.nnet.predict(canonicalBoard)
            # if local_history[s] > 1:
            #     return -v[0]

            # ps =np.array([1.0 for x in range(self.game.getActionSize())])
            # ps  = 0.1 * abs(np.random.randn(self.game.getActionSize()))
            # v = [abs(np.random.randn() * 0.1)]

            validMoves = self.game.getValidMoves(canonicalBoard, 1)
            ps = ps * validMoves  # masking invalid moves
            sum_Ps_s = np.sum(ps)
            if sum_Ps_s > 0:
                ps /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                print(canonicalBoard)
                print(ps)
                print(validMoves)
                ps = ps + validMoves
                ps /= np.sum(ps)
            self.Ps[s] = ps
            self.Vs[s] = validMoves
            self.Ns[s] = 0

            # print(".", end=" ")

            return -v[0]

        valids = self.Vs[s]

        valid_actions = list(filter(lambda a: valids[a] == 1, range(self.game.getActionSize())))
        # a = valid_actions[0]#choose the first (possibly only) at the beginning
        # skip actions that would lead to a repeated state...
        while True:
            a = self.getBestAction(s, valid_actions)
            next_board, next_player = self.game.getNextState(canonicalBoard, 1, a)
            next_s = next_board.getCanonicalForm(next_player)
            break
            # #if states keep repeating
            # if str(next_s) not in local_history:
            #     local_history[str(next_s)] = 1
            #     break
            # else:
            #     if local_history[str(next_s)] > 1:
            #         #try another action
            #         if len(valid_actions) > 1:
            #             valid_actions.remove(a)
            #             # print(f"Iter {self.search_counter}. Removed {a}. Remaining {valid_actions}. ")
            #             self.Qsa[(s, a)] = -1.0
            #             if not (s, a) in self.Nsa:
            #                 self.Nsa[(s, a)] = 0
            #         else:
            #             #trying to remove
            #             print (f"trying to remove {a}")
            #             break
            #     else:
            #         local_history[str(next_s)] += 1
            #         break
        v = self.search(next_s, local_history)

        self.deep -= 1
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = ((self.Nsa[(s, a)]) * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v


class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game, version, args):
        self.args = args
        self.action_size = game.getActionSize()
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
            Conv2D(args.num_channels, 2)(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv11 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 1)(h_conv1)))  # batch_size  x board_x x board_y x num_channels

        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 2)(h_conv11)))  # batch_size  x board_x x board_y x num_channels
        h_conv21 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 1)(h_conv2)))  # batch_size  x board_x x board_y x num_channels

        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 2)(h_conv21)))  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv31 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 1)(h_conv3)))  # batch_size  x (board_x) x (board_y) x num_channels

        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 2)(h_conv31)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv41 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 1)(h_conv4)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv41)

        s_fc1 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           loss_weights=[1., 5.], optimizer=Adam(args.lr))
        print(self.model.summary())

    def InitVersion37(self):
        x_image = BatchNormalization(axis=3)(
            Reshape((self.board_x, self.board_y, 4))(self.input_boards))  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same')(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same')(h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(
            h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(
            h_conv3)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           loss_weights=[1., 5.], optimizer=Adam(args.lr))
        print(self.model.summary())

    def evaluate(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
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

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """

        board = board.internalArray[np.newaxis, :, :]
        pi, v = self.model.predict(board)
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


class Coach():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = None
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        # board.display(1, self.mcts)
        NRs = {}

        while True:
            episodeStep += 1
            canonicalBoard = board.getCanonicalForm(self.curPlayer)
            temperature: int = int(episodeStep < self.args.tempThreshold)
            # temp = 1

            counts = self.mcts.getActionProb(canonicalBoard, temperature)
            retry = False
            allLegalMovesUsed = False
            draw = False

            while True:
                if temperature == 0:
                    bestA = np.argmax(counts)
                    probs = [0] * len(counts)
                    probs[bestA] = 1
                else:
                    counts = [x ** (1. / temperature) for x in counts]
                    probs = [x / float(sum(counts)) for x in counts]

                sym = self.game.getSymmetries(canonicalBoard, probs)

                action = np.random.choice(len(probs), p=probs)
                # if retry:
                #     print(f"Retry action {action}")
                new_board, new_Player = self.game.getNextState(board, self.curPlayer, action)

                s = str(new_board.getCanonicalForm(new_Player))
                if s not in NRs:
                    NRs[s] = 1
                else:
                    NRs[s] += 1
                if episodeStep > 300:
                    return []
                if NRs[s] < 4:
                    for b, p in sym:
                        trainExamples.append([b.internalArray, self.curPlayer, p, None])
                    break
                else:
                    # remove action from list and retry next action
                    counts[action] = 0
                    # print(f"Action {action} tried too many times. Retry...")
                    xxx = list(filter(lambda x: counts[x] != 0, [i for i in range(self.game.getActionSize())]))
                    # print(xxx)
                    # result = list(
                    #     filter(lambda p: self.isInAMill(board, p, -player) is False, available_opponent_pieces))
                    # result = [Game.validPositions.index(x) for x in result]
                    if np.sum(counts) == 0:
                        other_moves = self.game.getValidMoves(board, self.curPlayer)
                        if other_moves is [] or allLegalMovesUsed:
                            return []
                        allLegalMovesUsed = True
                        # print("Retry all legal moves")
                        # print(other_moves)
                        xxx = list(filter(lambda x: other_moves[x] != 0, [i for i in range(self.game.getActionSize())]))
                        # print(xxx)
                        if len(xxx) == 1:
                            break
                        counts = other_moves
                    else:
                        retry = True
                    # print("Draw")
                    # draw = True
                    # break

            board = new_board
            self.curPlayer = new_Player
            s = str(canonicalBoard)
            # print(str(episodeStep) + ": " + str(board.getPlayerCount()) + " - " + str(
            #     board.getOpponentCount()))
            # if board.getBoardStatus() == 0:
            # board.display(1, self.mcts)
            if (s, action) in self.mcts.Qsa:
                # print(str(episodeStep) + ": " + str(self.mcts.Qsa[(s, action)]) + " - " + str(board))
                print(
                    f"{episodeStep:003d}: {self.mcts.Qsa[(s, action)] * (-self.curPlayer):+4.2f} : {action:02d} : {board}")
                dummy = 0
            r = self.game.getGameEnded(board, self.curPlayer)
            if draw:
                r = 0.001  # draw
            # if episodeStep > 3000:
            #     r = 0
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def PitAgainst(self, neuralDataFileNumber):

        # self.pnet = self.nnet.__class__(self.game, args)  # the competitor network
        if self.pnet is None:
            self.pnet = NeuralNet(g, 37, args)
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename_no=neuralDataFileNumber)
        pmcts = MCTS(self.game, self.pnet, self.args)
        nmcts = MCTS(self.game, self.nnet, self.args)

        print(f'PITTING AGAINST {neuralDataFileNumber}')
        arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, 0)),
                      lambda x: np.argmax(pmcts.getActionProb(x, 0)), self.game, self.args)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompare, verbose=True)
        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

    def test(self):
        # test against a random

        otherPlayer = RandomPlayer(g).play
        mcts = MCTS(g, n, args)
        neuralPlayer = lambda x: np.argmax(mcts.getActionProb(x, 0))
        a = Arena(neuralPlayer, otherPlayer, g, args, mcts)
        result = a.playGames(2, verbose=False)

        # trainExamples = []
        # for e in a.trainExamplesHistory:
        #     trainExamples.extend(e)
        # for i in range(args.numIters // 10):
        #     print(f"ITERATION {i}")
        #     shuffle(trainExamples)
        #     n.train(trainExamples)
        #     n.save_checkpoint(folder= args.checkpoint, filename='new.neuralnet.data')
        print(result)
        pass

    def learn(self):
        # # how many iterations
        # for i in range(5):
        #     iterationTrainExamples = []
        #     for episodes in range(5):
        #         iterationTrainExamples += executeEpisode()
        #         pass
        #     pass
        # pass

        for i in range(0, self.args.numIterations + 1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for episode in range(self.args.numEpisodes):
                print(f"----- Episode {episode} -----")
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                example = self.executeEpisode()
                if example != []:
                    iterationTrainExamples += example
            # self.nnet.evaluate(iterationTrainExamples)
            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            print(f"len(trainExamplesHistory) ={len(self.trainExamplesHistory)}")
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            if trainExamples != []:
                self.nnet.train(trainExamples)

                # test against the best no36
                if i % 5 == 0:
                    # self.PitAgainst('no36.neural.data-ITER-390')
                    # self.PitAgainst(args.filename - 1)

                    # function
                    # otherPlayer = HumanPlayer(g).play
                    otherPlayer = RandomPlayer(g).play
                    # mcts = MCTS(g,n, args)
                    neuralPlayer = lambda x: np.argmax(self.mcts.getActionProb(x, temperature = 0))
                    a = Arena(neuralPlayer, otherPlayer, g, args, self.mcts)
                    result = a.playGames(10, verbose=True)
                    #
                    # trainExamples = []
                    # for e in a.trainExamplesHistory:
                    #     trainExamples.extend(e)
                    # for i in range(args.numIters // 10):
                    #     print(f"ITERATION {i}")
                    #     shuffle(trainExamples)
                    #     n.train(trainExamples)
                    #     n.save_checkpoint(folder= args.checkpoint, filename='new.neuralnet.data')
                    print(result)
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename_no=args.filename)

            # self.test()


    def saveTrainExamples(self, iteration):
        # folder = self.args.checkpoint
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        # with open(filename, "wb+") as f:
        #     Pickler(f).dump(self.trainExamplesHistory)
        # f.closed
        pass


    def loadTrainExamples(self):
        # modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        # examplesFile = modelFile+".examples"
        # if not os.path.isfile(examplesFile):
        #     print(examplesFile)
        #     r = input("File with trainExamples not found. Continue? [y|n]")
        #     if r != "y":
        #         sys.exit()
        # else:
        #     print("File with trainExamples found. Read it.")
        #     with open(examplesFile, "rb") as f:
        #         self.trainExamplesHistory = Unpickler(f).load()
        #     f.closed
        #     # examples based on the model were already collected (loaded)
        #     self.skipFirstSelfPlay = True
        pass


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)

        # if pass is required
        if valid[88] == 1:
            return 88
        # for i in range(len(valid)):
        #     if valid[i]:
        #         print(int(i/self.game.n), int(i%self.game.n))
        while True:
            input_string = input()
            try:
                input_array = [int(x) for x in input_string.split(' ')]
            except:
                input_array = []
            if len(input_array) == 2:
                a, b = input_array
                try:
                    move = Game.validPositions.index((a, b))
                except:
                    move = 90
            elif len(input_array) == 4:
                a, b, c, d = input_array
                try:
                    move = 24 + Game.validActions.index(((a, b), (c, d)))
                except:
                    move = 90
            else:
                move = 90  # invalid

            if move <= 88 and valid[move]:
                break
            else:
                print('Invalid')

        return move


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


args = dotdict({
        'numIterations': 1000,
        'numEpisodes': 10,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSimulations': 30,
        'arenaCompare': 9,
        'cpuct': 1,
        'checkpoint': './temp/',
        'load_model': False,
        # 'filename' : 'no27.neural.data',#2-2-2-2-2
        # 'filename': 'no28.neural.data',  # 3-3-3-3-3
        # 'filename': 'no32.neural.data',  # 1

        # 'filename' : 'no35.neural.data',#2-2-2-2-2
        'filename': 'Moara6NoExtra',

        # 'filename': 'no37.neural.data',  # 1
        'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
        'numItersForTrainExamplesHistory':50,

        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 20,
        'batch_size': 64,
        'cuda': True,
        'num_channels': 512,

    })

if __name__ == "__main__":

    g = Game()
    n = NeuralNet(g, 0, args)
    n.load_checkpoint(folder=args.checkpoint, filename_no=args.filename)

    # play
    # function
    # otherPlayer = HumanPlayer(g).play
    # # otherPlayer = RandomPlayer(g).play
    # mcts = MCTS(g,n, args)
    # neuralPlayer = lambda x: np.argmax(mcts.getActionProb(x, temp = 0))
    # a = Arena(neuralPlayer, otherPlayer, g, args, mcts)
    # result = a.playGames(10, verbose=True)
    #
    # trainExamples = []
    # for e in a.trainExamplesHistory:
    #     trainExamples.extend(e)
    # for i in range(args.numIters // 10):
    #     print(f"ITERATION {i}")
    #     shuffle(trainExamples)
    #     n.train(trainExamples)
    #     n.save_checkpoint(folder= args.checkpoint, filename='new.neuralnet.data')
    # print(result)

    # train
    c = Coach(g, n, args)
    c.learn()

    # m = MCTS2(g, n, args)
    # m.policyIterSP()
    print("moara")
