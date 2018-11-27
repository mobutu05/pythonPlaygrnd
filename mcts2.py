import keras
import numpy as np
import moara


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


class MoaraBoard(IBoard):
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


# interface for game classes
class IGame:
    def getInitBoard(self) -> IBoard:
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass


# instance of a IGame
class Moara(IGame):
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return MoaraBoard()


class MCTS:
    def __init__(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times action a was taken from board s
        self.Ns = {}  # stores #times board s
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard: IBoard, temperature: float = 1):
        pass


def executeEpisode(game: IGame, mcts: MCTS):
    board = game.getInitBoard()
    currentPlayer = 1  # alternate between 1(white) and -1(black)
    episodeStep = 0
    while True:
        episodeStep += 1
        canonicalBoard = board.getCanonicalForm(currentPlayer)
        temperature: int = int(episodeStep < moara.args.tempThreshold)
        mcts.getActionProbabilities(canonicalBoard, temperature)
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
