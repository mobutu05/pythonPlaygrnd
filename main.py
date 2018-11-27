from keras import Input, Model
from keras.layers import Reshape, Activation, BatchNormalization, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from collections import deque
# from progress.bar import Bar
import time
import numpy as np
import math
from random import shuffle
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value
#        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
#     """
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
# class DotDict(dict):
#     def __getattr__(self, name):
#         return self[name]
#
#
# args = DotDict({
#     # 'numIters': 1000,
#     'numEps': 100,
#     'tempThreshold': 15,
#     # 'updateThreshold': 0.6,
#     'maxlenOfQueue': 200000,
#     'numMCTSSims': 25,
#     # 'arenaCompare': 40,
#     'cpuct': 1,
#     #
#     # 'checkpoint': './temp/',
#     # 'load_model': False,
#     # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
#     # 'numItersForTrainExamplesHistory': 20,
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': True,
#     'num_channels': 512,
#     'numIters': 1000,
# })
#
# class Board():
#
#     # list of all 8 directions on the board, as (x,y) offsets
#     __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]
#
#     def __init__(self, n=3):
#         "Set up initial board configuration."
#
#         self.n = n
#         # Create the empty board array.
#         self.pieces = [None]*self.n
#         for i in range(self.n):
#             self.pieces[i] = [0]*self.n
#
#     # add [][] indexer syntax to the Board
#     def __getitem__(self, index):
#         return self.pieces[index]
#
#     def is_win(self, color):
#         """Check whether the given player has collected a triplet in any direction;
#         @param color (1=white,-1=black)
#         """
#         win = self.n
#         # check y-strips
#         for y in range(self.n):
#             count = 0
#             for x in range(self.n):
#                 if self[x][y] == color:
#                     count += 1
#             if count == win:
#                 return True
#         # check x-strips
#         for x in range(self.n):
#             count = 0
#             for y in range(self.n):
#                 if self[x][y] == color:
#                     count += 1
#             if count == win:
#                 return True
#         # check two diagonal strips
#         count = 0
#         for d in range(self.n):
#             if self[d][d] == color:
#                 count += 1
#         if count == win:
#             return True
#         count = 0
#         for d in range(self.n):
#             if self[d][self.n - d - 1] == color:
#                 count += 1
#         if count == win:
#             return True
#
#         return False
#
#     def has_legal_moves(self):
#         for y in range(self.n):
#             for x in range(self.n):
#                 if self[x][y]==0:
#                     return True
#         return False
#
#     def get_legal_moves(self, color):
#         """Returns all the legal moves for the given color.
#         (1 for white, -1 for black)
#         @param color not used and came from previous version.
#         """
#         moves = set()  # stores the legal moves.
#
#         # Get all the empty squares (color==0)
#         for y in range(self.n):
#             for x in range(self.n):
#                 if self[x][y]==0:
#                     newmove = (x,y)
#                     moves.add(newmove)
#         return list(moves)
#
#     def execute_move(self, move, color):
#         """Perform the given move on the board;
#         color gives the color pf the piece to play (1=white,-1=black)
#         """
#
#         (x,y) = move
#
#         # Add the piece to the empty square.
#         assert self[x][y] == 0
#         self[x][y] = color
#
# class Game:
#     def __init__(self):
#         self.n = 3
#
#     def getBoardSize(self):
#         # (a,b) tuple
#         return self.n, self.n
#
#     def getActionSize(self):
#         # return number of actions
#         return self.n * self.n + 1
#
#     def getCanonicalForm(self, board, player):
#         # return state if player==1, else return -state if player==-1
#         return player*board
#
#     def getInitBoard(self):
#         # return initial board (numpy board)
#         b = Board(self.n)
#         return np.array(b.pieces)
#
#     def stringRepresentation(self, board):
#         # 8x8 numpy array (canonical board)
#         hh = ''
#         for y in range(self.n):
#             for x in range(self.n):
#                hh = hh + str(board[x][y])
#         return hh
#
#     def getGameEnded(self, board, player):
#         # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
#         # player = 1
#         b = Board(self.n)
#         b.pieces = np.copy(board)
#
#         if b.is_win(player):
#             return 1
#         if b.is_win(-player):
#             return -1
#         if b.has_legal_moves():
#             return 0
#         # draw has a very little value
#         return 1e-4
#
#     def getValidMoves(self, board, player):
#         # return a fixed size binary vector
#         valids = [0]*self.getActionSize()
#         b = Board(self.n)
#         b.pieces = np.copy(board)
#         legalMoves =  b.get_legal_moves(player)
#         if len(legalMoves)==0:
#             valids[-1]=1
#             return np.array(valids)
#         for x, y in legalMoves:
#             valids[self.n*x+y]=1
#         return np.array(valids)
#
#     def getNextState(self, board, player, action):
#         # if player takes action on board, return next (board,player)
#         # action must be a valid move
#         if action == self.n*self.n:#no move, passing
#             return (board, -player)
#         b = Board(self.n)
#         b.pieces = np.copy(board)
#         move = (int(action/self.n), action%self.n)
#         b.execute_move(move, player)
#         return (b.pieces, -player)
#
#     def getSymmetries(self, board, pi):
#         # mirror, rotational
#         assert(len(pi) == self.n**2+1)  # 1 for pass
#         pi_board = np.reshape(pi[:-1], (self.n, self.n))
#         l = []
#
#         for i in range(1, 5):
#             for j in [True, False]:
#                 newB = np.rot90(board, i)
#                 newPi = np.rot90(pi_board, i)
#                 if j:
#                     newB = np.fliplr(newB)
#                     newPi = np.fliplr(newPi)
#                 l += [(newB, list(newPi.ravel()) + [pi[-1]])]
#         return l
#
# class NeuralNet:
#     def __init__(self, game):
#         # game params
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()
#         # Neural Net
#         self.input_boards = Input(shape=(self.board_x, self.board_y))
#         x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1
#         h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
#             Conv2D(args.num_channels, 3, padding='same')(x_image)))  # batch_size  x board_x x board_y x num_channels
#         h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
#             Conv2D(args.num_channels, 3, padding='same')(h_conv1)))  # batch_size  x board_x x board_y x num_channels
#         h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(
#             h_conv2)))  # batch_size  x (board_x) x (board_y) x num_channels
#         h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(
#             h_conv3)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
#         h_conv4_flat = Flatten()(h_conv4)
#         s_fc1 = Dropout(args.dropout)(
#             Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
#         s_fc2 = Dropout(args.dropout)(
#             Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024
#         self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
#         self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1
#
#         self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
#         self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
#
#     def predict(self, board):
#         """
#         board: np array with board
#         """
#         # timing
#         start = time.time()
#
#         # preparing input
#         board = board[np.newaxis, :, :]
#
#         # run
#         # pi, v = self.model.predict(board)
#         pi = [1.0]*self.action_size
#         v = 0
#
#         #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
#         # return pi[0], v[0]
#         return pi, v
#
# class Coach:
#     def __init__(self, game, nnet):
#         self.game = game
#         self.nnet = nnet
#         self.pnet = self.nnet.__class__(self.game)  # the competitor network
#         self.mcts = MCTS(self.game, self.nnet)
#         # history of examples from args.numItersForTrainExamplesHistory latest iterations
#         self.trainExamplesHistory = []
#         # can be overriden in loadTrainExamples()
#         self.skipFirstSelfPlay = False
#
#     def executeEpisode(self):
#         trainExamples = []
#         board = self.game.getInitBoard()
#         self.curPlayer = 1
#         episodeStep = 0
#         while True:
#             episodeStep += 1
#             canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
#             temp = int(episodeStep < args.tempThreshold)
#             p_i = self.mcts.getActionProb(canonicalBoard, temp=temp)
#             sym = self.game.getSymmetries(canonicalBoard, p_i)
#             for b,p in sym:
#                 trainExamples.append([b, self.curPlayer, p, None])
#
#             action = np.random.choice(len(p_i), p=p_i)
#             board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
#
#             r = self.game.getGameEnded(board, self.curPlayer)
#
#             if r!=0:
#                 return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]
#             pass
#         pass
#
#     def learn(self):
#         for i in range(1, args.numIters + 1):
#             # bookkeeping
#             print('------ITER ' + str(i) + '------')
#             if not self.skipFirstSelfPlay or i > 1:
#                 iterationTrainExamples = deque([], maxlen=args.maxlenOfQueue)
#
#                 eps_time = AverageMeter()
#                 # bar = Bar('Self Play', max= args.numEps)
#                 end = time.time()
#
#                 for eps in range(args.numEps):
#                     self.mcts = MCTS(self.game, self.nnet)  # reset search tree
#                     iterationTrainExamples += self.executeEpisode()
#
#                     # bookkeeping + plot progress
#                     eps_time.update(time.time() - end)
#                     end = time.time()
#                 #     bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
#                 #         eps=eps + 1, maxeps=args.numEps, et=eps_time.avg,
#                 #         total=bar.elapsed_td, eta=bar.eta_td)
#                 #     bar.next()
#                 # bar.finish()
#                 # save the iteration examples to the history
#                 self.trainExamplesHistory.append(iterationTrainExamples)
#
#             if len(self.trainExamplesHistory) > args.numItersForTrainExamplesHistory:
#                 print("len(trainExamplesHistory) =", len(self.trainExamplesHistory),
#                       " => remove the oldest trainExamples")
#                 self.trainExamplesHistory.pop(0)
#             # backup history to a file
#             # NB! the examples were collected using the model from the previous iteration, so (i-1)
#             self.saveTrainExamples(i - 1)
#
#             # shuffle examlpes before training
#             trainExamples = []
#             for e in self.trainExamplesHistory:
#                 trainExamples.extend(e)
#             shuffle(trainExamples)
#
#             # training new network, keeping a copy of the old one
#             self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
#             self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
#             pmcts = MCTS(self.game, self.pnet, self.args)
#
#             self.nnet.train(trainExamples)
#             nmcts = MCTS(self.game, self.nnet, self.args)
#
#             print('PITTING AGAINST PREVIOUS VERSION')
#             arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
#                           lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
#             pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
#
#             print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
#             if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.args.updateThreshold:
#                 print('REJECTING NEW MODEL')
#                 self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
#             else:
#                 print('ACCEPTING NEW MODEL')
#                 self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
#                 self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
#
#
# class MCTS:
#     def __init__(self, game, nnet):
#         self.game = game
#         self.nnet = nnet
#         self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
#         self.Nsa = {}  # stores #times edge s,a was visited
#         self.Ns = {}  # stores #times board s was visited
#         self.Ps = {}  # stores initial policy (returned by neural net)
#
#         self.Es = {}  # stores game.getGameEnded ended for board s
#         self.Vs = {}  # stores game.getValidMoves for board s
#
#     def getActionProb(self, canonicalBoard, temp=1):
#         """
#         This function performs numMCTSSims simulations of MCTS starting from
#         canonicalBoard.
#
#         Returns:
#             probs: a policy vector where the probability of the ith action is
#                    proportional to Nsa[(s,a)]**(1./temp)
#         """
#         for i in range(args.numMCTSSims):
#             self.search(canonicalBoard)
#         s = self.game.stringRepresentation(canonicalBoard)
#         counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
#
#         if temp == 0:
#             bestA = np.argmax(counts)
#             probs = [0] * len(counts)
#             probs[bestA] = 1
#             return probs
#
#         counts = [x ** (1. / temp) for x in counts]
#         probs = [x / float(sum(counts)) for x in counts]
#         return probs
#
#     def search(self, canonicalBoard):
#         """
#         This function performs one iteration of MCTS. It is recursively called
#         till a leaf node is found. The action chosen at each node is one that
#         has the maximum upper confidence bound as in the paper.
#
#         Once a leaf node is found, the neural network is called to return an
#         initial policy P and a value v for the state. This value is propogated
#         up the search path. In case the leaf node is a terminal state, the
#         outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
#         updated.
#
#         NOTE: the return values are the negative of the value of the current
#         state. This is done since v is in [-1,1] and if v is the value of a
#         state for the current player, then its value is -v for the other player.
#
#         Returns:
#             v: the negative of the value of the current canonicalBoard
#         """
#
#         s = self.game.stringRepresentation(canonicalBoard)
#
#         if s not in self.Es:
#             end = self.game.getGameEnded(canonicalBoard, 1)
#             self.Es[s] = end
#         if self.Es[s]!=0:
#             # terminal node
#             return -self.Es[s]
#
#         if s not in self.Ps:
#             # leaf node
#             policy, v_ = self.nnet.predict(canonicalBoard)
#             valids = self.game.getValidMoves(canonicalBoard, 1)
#             policy = policy*valids      # masking invalid moves
#             sum = np.sum(policy)
#             policy /=  sum   # renormalize
#             self.Ps[s] = policy
#             self.Vs[s] = valids
#             self.Ns[s] = 0
#             return -v_
#
#         valids = self.Vs[s]
#         cur_best = -float('inf')
#         best_act = -1
#
#         # pick the action with the highest upper confidence bound
#         for a in range(self.game.getActionSize()):
#             if valids[a]:
#                 if (s,a) in self.Qsa:
#                     u = self.Qsa[(s,a)] + args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
#                 else:
#                     u = args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])     # Q = 0 ?
#
#                 if u > cur_best:
#                     cur_best = u
#                     best_act = a
#
#         a = best_act
#         next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
#         next_s = self.game.getCanonicalForm(next_s, next_player)
#
#         v = self.search(next_s)
#
#         if (s,a) in self.Qsa:
#             qsa = self.Qsa[(s,a)]
#             nsa = self.Nsa[(s,a)]
#             qsa = (nsa * qsa + v)/(self.Nsa[(s,a)]+1)
#
#             nsa += 1
#             self.Nsa[(s,a)] = nsa
#             self.Qsa[(s, a)] = qsa
#
#         else:
#             self.Qsa[(s,a)] = v
#             self.Nsa[(s,a)] = 1
#
#         ns = self.Ns[s]
#         ns += 1
#         self.Ns[s] = ns
#         return -v


if __name__ == "__main__":
    # g = Game()
    # nn = NeuralNet(g)
    # c = Coach(g, nn)
    # c.learn()
    a: int = 5
    b: float = 1.3
    c: float = a + b
    print(c)
    pass
