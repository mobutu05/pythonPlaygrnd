import keras
import numpy as np
import moara
import copy


# interface for board of a game
class IBoard:
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
        pass


# interface for game classes
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


# instance of IBoard
class MoaraBoard(IBoard):

    def getCanonicalForm(self, player) -> IBoard:
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
            return MoaraBoard(self.internalArray)
        else:
            b = MoaraBoard(np.array([self.internalArray[0] * player, self.internalArray[2], self.internalArray[1],
                                     self.internalArray[3] * player]))
            return b


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
        hh = hh + str(self.getOpponentCount(1))  # opponent
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

    def getPosition(self, pos):
        (x, y) = pos
        return self.internalArray[0][y][x]

    def getBoardStatus(self):
        return int(round(self.internalArray[3][3][3]))

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

    #
    # def getOpponentCount(self, player):
    #     no_pieces_on_board = len([0 for pos in self.VALID_POSITIONS if self.getPosition(pos) == -player])
    #     if player == 1:
    #         return int(round(self.internalArray[2][3][3]))
    #     else:
    #         return int(round(self.internalArray[1][3][3]))

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

    # list of legal moves from the current position for the player
    def getLegalMovesList(self, player):
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
        player_valid_moves_list = self.getLegalMovesList(player)
        opponent_valid_moves_list = self.getLegalMovesList(-player)
        if player_valid_moves_list == [] and opponent_valid_moves_list == []:
            return 0.001  # draw
        if self.getOpponentCount(player) < 3 or opponent_valid_moves_list == []:
            return 1
        if self.getPlayerCount(player) < 3 or player_valid_moves_list == []:
            return -1
        return 0


class MCTS:
    def __init__(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times action a was taken from board s
        self.Ns = {}  # stores #times board s
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Vs = {}  # stores game.getValidMoves for board s

    def search(self, game: IGame):
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
        e = game.getGameEnded()
        pass

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
                v = self.search(game)

        s = str(game)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

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
mcts = MCTS()
learn(moaraGame, mcts)
