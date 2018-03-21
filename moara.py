import os
import numpy as np
import functools
import keras
from keras import Input, Model
from keras.layers import Reshape, Activation, Conv2D, BatchNormalization, Flatten, Dense, Dropout
from keras.optimizers import Adam
import math

model = None


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


def getInitBoard():
    arr = np.array([[0 for y in range(7)] for x in range(7)])
    arr[3][3] = 99  # player/opponent
    return arr


def getCanonicalForm(board, player):
    b = player * board
    if player == -1:
        # switch digits - first one is crt player
        n = - b[3][3]
        b[3][3] = (n // 10) + (n % 10) * 10
    return b


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
    ((1, 1), (1, 3)), ((1, 3), (1, 5)),
    ((2, 2), (2, 3)), ((2, 3), (2, 4)),
    ((3, 0), (3, 1)), ((3, 1), (3, 2)),
    ((3, 4), (3, 5)), ((3, 5), (3, 6)),
    ((4, 2), (4, 3)), ((4, 3), (4, 4)),
    ((5, 1), (5, 3)), ((5, 3), (5, 5)),
    ((6, 0), (6, 3)), ((6, 3), (6, 6)),
    # vertical
    ((0, 0), (3, 0)), ((3, 0), (6, 0)),
    ((1, 1), (3, 1)), ((3, 1), (5, 1)),
    ((2, 2), (3, 2)), ((3, 2), (4, 2)),
    ((0, 3), (1, 3)), ((1, 3), (2, 3)),
    ((4, 3), (5, 3)), ((5, 3), (6, 3)),
    ((2, 4), (3, 4)), ((3, 4), (4, 4)),
    ((1, 5), (3, 5)), ((3, 5), (5, 5)),
    ((0, 6), (3, 6)), ((3, 6), (6, 6))
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
    ((4, 3), (4, 3), (6, 3)),
    ((2, 4), (3, 4), (4, 4)),
    ((1, 5), (3, 5), (5, 5)),
    ((0, 6), (3, 6), (6, 6))
]


def getActionSize():
    # return number of actions
    return  (24 +  #fly 1st piece
            24 + #fly 2nd piece
            24 + #fly 3rd piece
            24 + #put piece
            24 + #capture piece
            32)   #move piece



def getPosition(board, pos):
    (x, y) = pos
    return board[x][y]


def isMill(board, mill, player):
    count = functools.reduce(lambda acc, i: 1 if getPosition(board, mill[i]) == player else 0, range(3), 0)
    return 1 if count == 3 else 0


# is the piece in a mill of a player?
def isInAMill(board, pos, player):
    # find all mills that contain the pos
    mill_list = list(filter(lambda mill: list(filter(lambda p: p == pos, mill)) != [], mills))
    return list(filter(lambda x: isMill(board, x, player) == 1, mill_list)) != []


def getLegalMoves(board, player):
    pieces_on_board = functools.reduce(lambda acc, pos: acc + (1 if getPosition(board, pos) == player else 0),
                                       validPositions, 0)
    result = []

    player_no = board[3][3] // 10
    if player_no > pieces_on_board:
        # phase 1: can put anywhere where there is an empty place
        result = list(filter(lambda x: getPosition(board, x) == 0, validPositions))
        result = [3 * 24 + validPositions.index(x) for x in result]
    elif player_no > 3:
        # phase 2: need to move a piece in a valid position
        # select all transitions that have a player piece in the first position and an empty one in the second position
        result = list(filter(lambda x: getPosition(board, x[0] == player) and
                                       getPosition(board, x[1] == 0), validActions))
        result = [5 * 24 + validPositions.index(x) for x in result]
    else:
        #phase 3: when only 3 pieces left can move anywhere empty

        empty = list(filter(lambda x: getPosition(board, x) == 0, validPositions))
        #first pieces
        result = [0 * 24 + validPositions.index(x) for x in empty]
        #second piece
        result += [1 * 24 + validPositions.index(x) for x in empty]
        #third piece
        result += [2 * 24 + validPositions.index(x) for x in empty]
    pass

    # if the player has a mill, it must remove an opponent's piece, that is not a mill either
    mill_no = functools.reduce(lambda acc, mill: isMill(board, mill, player_no), mills, 0)

    # need to select an opponent piece
    if mill_no > 0:
        available_opponent_pieces = list(filter(lambda x: getPosition(board, x) == -player, board))
        result = list(filter(lambda p: isInAMill(board, p, -player) is False, available_opponent_pieces))
        result += [4 * 24 + validPositions.index(x) for x in result]

    return result

def getNextState(canonicalBoard, player, a):
    # if player takes action on board, return next (board,player)
    # action must be a valid move

    #phase
    category = a // 24
    #phase 3
    if category >= 0 and category <= 2:
        pass
    #phase 1
    if category == 3:
        (x,y) = validPositions[a % 24]
        canonicalBoard[x][y] = player
        return (canonicalBoard, -player)
        pass
    #capture
    if category == 4:
        pass
    #move
    if category > 4:
        pass
    # if a[0] == (-1, -1):
    #
    # if action == self.n * self.n:
    #     return (board, -player)
    # b = Board(self.n)
    # b.pieces = np.copy(board)
    # move = (int(action / self.n), action % self.n)
    # b.execute_move(move, player)
    # return (b.pieces, -player)
    pass

def getValidMoves(game, player):
    # make it a 0 and 1 array for each legal move
    moves = getLegalMoves(game, player)
    #return list(map(lambda x: 1 if x in moves else 0, validActions))
    return [1 if x in moves else 0 for x in range(getActionSize())]


def getGameEnded(board, player):
    # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
    # player win if:
    #   - opponent has less than 3 pieces
    #   - opponent has no valid moves
    # draw if:
    #   - no valid moves for any player
    #   - future:
    #       - 50 moves with no capture
    #       - position replay 3 times
    player_pieces = board[3][3] // 10
    opponent_pieces = board[3][3] % 10
    player_valid = getLegalMoves(board, player)
    opponent_valid = getLegalMoves(board, -player)
    if player_valid == [] and opponent_valid == []:
        return 0.5  # draw
    if opponent_pieces < 3 or opponent_valid == []:
        return 1
    if player_pieces < 3 or player_valid == []:
        return -1
    return 0


def toString(board):
    hh = ''
    for (x, y) in validPositions:
        hh = hh + str(board[y][x])
    hh = hh + str(board[3][3] // 10)
    hh = hh + str(board[3][3] % 10)
    return hh


EndGames = {}  # stores game.getGameEnded ended for board s
initialPolicies = {}  # stores initial policy (returned by neural net)
ValidMoves = {}  # stores game.getValidMoves for board s
Ns = {}        # stores #times board s was visited
Qsa = {}       # stores Q values for s,a (as defined in the paper)
Nsa = {}       # stores #times action a was taken from state s

def MCTSearch(canonicalBoard):
    s = toString(canonicalBoard)
    if s not in EndGames:
        EndGames[s] = getGameEnded(canonicalBoard, 1)
    if EndGames[s] != 0:
        # terminal node
        return -EndGames[s]

    if s not in initialPolicies:
        # leaf node
        initialPolicies[s], v = predict(canonicalBoard)
        valids = getValidMoves(canonicalBoard, 1)
        initialPolicies[s] = initialPolicies[s] * valids  # masking invalid moves

        sum = np.sum(initialPolicies[s])
        if sum > 0:
            initialPolicies[s] /= sum  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            print("All valid moves were masked, do workaround.")
            initialPolicies[s] = initialPolicies[s] + valids
            initialPolicies[s] /= np.sum(initialPolicies[s][s])

        ValidMoves[s] = valids
        Ns[s] = 0
        return -v
    #if initial policy already evaluated
    valids = ValidMoves[s]
    cur_best = -float('inf')
    best_act = []

    # pick the action with the highest upper confidence bound
    cpuct = 1
    EPS = 1e-8
    for a in range(getActionSize()):
        if valids[a]:
            if (s, a) in Qsa:
                u = Qsa[(s, a)] + cpuct * initialPolicies[s][a] * math.sqrt(Ns[s]) / (1 + Nsa[(s, a)])
            else:
                u = cpuct * initialPolicies[s][a] * math.sqrt(Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = [a]
            elif u == cur_best:
                best_act.append(a)

    a = np.random.choice(best_act)
    # a = best_act
    next_s, next_player = getNextState(canonicalBoard, 1, a)
    next_s = getCanonicalForm(next_s, next_player)

    v = MCTSearch(next_s)

    if (s, a) in Qsa:
        Qsa[(s, a)] = (Nsa[(s, a)] * Qsa[(s, a)] + v) / (Nsa[(s, a)] + 1)
        Nsa[(s, a)] += 1

    else:
        Qsa[(s, a)] = v
        Nsa[(s, a)] = 1

    Ns[s] += 1
    return -v

def NNInit():
    global model
    num_channels = 512
    dropout = 0.3
    action_size = getActionSize()
    board_x = 7
    board_y = 7
    input_boards = Input(shape=(board_x, board_y))  # s: batch_size x board_x x board_y
    x_image = Reshape((board_x, board_y, 1))(input_boards)  # batch_size  x board_x x board_y x 1
    h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
        Conv2D(num_channels, 3, padding='same')(x_image)))  # batch_size  x board_x x board_y x num_channels
    h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
        Conv2D(num_channels, 3, padding='same')(h_conv1)))  # batch_size  x board_x x board_y x num_channels
    h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
        Conv2D(num_channels, 3, padding='same')(h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
    h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='valid')(
        h_conv3)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
    h_conv4_flat = Flatten()(h_conv4)
    s_fc1 = Dropout(dropout)(
        Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
    s_fc2 = Dropout(dropout)(
        Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
    pi = Dense(action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
    v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1
    lr = 0.001
    model = Model(inputs=input_boards, outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr))
    pass


def predict(board):
    global model
    board = board[np.newaxis, :, :]
    pi, v = model.predict(board)
    return pi[0], v[0]


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


NNInit()
loadNN()
learn()
print("moara")
