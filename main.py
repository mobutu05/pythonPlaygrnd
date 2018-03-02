from keras import Input, Model
from keras.layers import Reshape, Activation, BatchNormalization, Conv2D, Flatten, Dropout, Dense
from keras.optimizers import Adam


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


args = DotDict({
    # 'numIters': 1000,
    # 'numEps': 100,
    # 'tempThreshold': 15,
    # 'updateThreshold': 0.6,
    # 'maxlenOfQueue': 200000,
    # 'numMCTSSims': 25,
    # 'arenaCompare': 40,
    # 'cpuct': 1,
    #
    # 'checkpoint': './temp/',
    # 'load_model': False,
    # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    # 'numItersForTrainExamplesHistory': 20,
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
    'numIters': 1000,
})


class Game:
    def __init__(self):
        self.n = 3

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1


class NeuralNet:
    def __init__(self, game):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1
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
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))


class Coach:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet)
        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory = []
        # can be overriden in loadTrainExamples()
        self.skipFirstSelfPlay = False

    def learn(self):
        for i in range(1, args.numIters + 1):
            pass


class MCTS:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s


if __name__ == "__main__":
    g = Game()
    nn = NeuralNet(g)
    c = Coach(g, nn)
    c.learn()
    pass
