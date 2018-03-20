import os
import numpy as np


def loadNN():
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join("./nndb.data")
    if not os.path.exists(filepath):
        print("No model in path '{}'".format(filepath))
    # self.nnet.model.load_weights(filepath)
    pass


def SaveNN():
    filepath = os.path.join("./nndb.data")
    # if not os.path.exists(folder):
    #     print("Checkpoint Directory does not exist! Making directory {}".format(folder))
    #     os.mkdir(folder)
    # else:
    #     print("Checkpoint Directory exists! ")
    # self.nnet.model.save_weights(filepath)
    pass


class Game():
    pass


def getInitBoard():
    return np.array([[0. for y in range(1, 7)] for x in range(1, 7)])


def getCanonicalForm(board, player):
    # return state if player==1, else return -state if player==-1
    return player * board


validPoints = [(0, 0), (0, 3), (0, 6), (1, 1), (1, 3), (1, 5), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 4),
               (3, 5), (3, 6), (4, 2), (4, 3), (4, 4), (5, 1), (5, 3), (5, 5), (6, 0), (6, 3), (6, 6)]
validMoves = [((0, 0), (0, 3)), ((0, 3), (0, 6)),
              ((1, 0), (1, 3)), ((1, 3), (1, 6)), ]


def getGameEnded(board, player):
    # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
    # player = 1

    for w in range(self.n):
        for h in range(self.n):
            if (w in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[i][h] for i in range(w, w + n))) == 1):
                return board[w][h]
            if (h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w][j] for j in range(h, h + n))) == 1):
                return board[w][h]
            if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w + k][h + k] for k in range(n))) == 1):
                return board[w][h]
            if (w in range(self.n - n + 1) and h in range(self.n - n + 1, self.n) and board[w][h] != 0 and
                        len(set(board[w + l][h - l] for l in range(n))) == 1):
                return board[w][h]
    if b.has_legal_moves():
        return 0
    return 1e-4


def toString(board):
    # 8x8 numpy array (canonical board)
    hh = ''
    for (x, y) in validPoints:
        hh = hh + str(board[y][x])
    return hh


Es = {}  # stores game.getGameEnded ended for board s


def MCTSearch(canonicalBoard):
    s = toString(canonicalBoard)
    if s not in Es:
        Es[s] = getGameEnded(canonicalBoard, 1)
    pass


def getActionProb(canonicalBoard, temp=1):
    for i in range(10):  # no of montecarlo simulations
        MCTSearch(canonicalBoard)
    pass


# one episode of self-play
def executeEpisode():
    board = getInitBoard()
    crtPlayer = 1
    steps = 0  # how many steps in this episode

    while True:
        steps += 1
        canonicalBoard = getCanonicalForm(board, crtPlayer)
        pi = getActionProb(canonicalBoard)
        pass
    pass


def learn():
    # how many iterations
    for i in range(1, 5):
        for episodes in range(1, 5):
            executeEpisode()
            pass
        pass
    pass


loadNN()
learn()
print("moara")
