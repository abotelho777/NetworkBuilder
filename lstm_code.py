import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
import time


class Network:
    def __init__(self):
        self.layers = []
        self.__is_init = False
        self.step_size = None
        self.batch_size = None
        self.training_epochs = None

        self.is_recurrent = False
        self.max_steps = 151

        self.args = dict()
        # self.session = tf.InteractiveSession()
        self.session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))  # use CPU for now

    def add_input_layer(self, n, normalize=True):
        layer = dict()
        layer['n'] = n
        layer['z'] = tf.placeholder(tf.float32, [1, self.max_steps, n], name='x')
        layer['param'] = {'w': None, 'b': None, 'type': 'input',
                          'arg': {'mean': tf.placeholder(tf.float32, name='input_mean'),
                                  'stdev': tf.placeholder(tf.float32, name='input_stdev')}}
        layer['a'] = tf.identity
        if normalize:
            layer['h'] = layer['a']((layer['z']-layer['param']['arg']['mean']) /
                                    tf.maximum(layer['param']['arg']['stdev'], tf.constant(1e-12, dtype=tf.float32)))
        else:
            layer['h'] = layer['a'](layer['z'])

        self.layers.insert(max(0, len(self.layers)), layer)
        self.args[self.layers[-1]['param']['arg']['mean']] = 0
        self.args[self.layers[-1]['param']['arg']['stdev']] = 1
        return self

    def add_dense_layer(self, n, activation=tf.identity):
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'dense', 'arg': None}
        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32)
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]))
        layer['z'] = tf.matmul(self.layers[-1]['h'], layer['param']['w']) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        return self

    def add_inverse_layer(self, layer_index, activation=tf.identity):
        if layer_index < 0:
            layer_index += len(self.layers)
        assert layer_index > 0 and layer_index < len(self.layers)
        inv = self.layers[layer_index]
        layer = dict()
        layer['n'] = self.layers[layer_index-1]['n']
        layer['param'] = {'w': None, 'b': None, 'type': 'inverse', 'arg': layer_index}
        layer['param']['w'] = None
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]))
        layer['z'] = tf.matmul(self.layers[-1]['h'], tf.transpose(inv['param']['w'])) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        return self

    def add_dropout_layer(self, n, keep=0.5, activation=tf.identity):
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'dropout',
                          'arg': tf.placeholder(tf.float32, name='keep')}

        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32)
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]))
        layer['z'] = tf.matmul(tf.nn.dropout(self.layers[-1]['h'], layer['param']['arg']),
                               layer['param']['w']) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        self.args[self.layers[-1]['param']['arg']] = keep
        return self

    def add_LSTM_layer(self,n,activation=tf.identity, use_last=False, dropout_keep=1.):
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'lstm',
                          'arg': {'cell': None, 'state': None, 'steps': tf.placeholder(tf.int32,name='steps'),
                                  'keep': tf.placeholder(tf.float32, name='keep')}}

        layer['param']['arg']['cell'] = tf_rnn.BasicLSTMCell(layer['n'], state_is_tuple=True)

        temp_z, layer['param']['arg']['state'] = tf.nn.dynamic_rnn(layer['param']['arg']['cell'],
                                                                        tf.reshape(self.layers[-1]['h'],
                                                                                   [1, self.max_steps,
                                                                                    self.layers[0]['n']]),
                                                                   dtype=tf.float32)
        layer['a'] = activation

        if use_last:
            temp_z = tf.transpose(temp_z, [1, 0, 2])
            layer['z'] = tf.gather(temp_z, layer['param']['arg']['steps']-1)
        else:
            layer['z'] = temp_z

        layer['h'] = tf.nn.dropout(layer['a'](layer['z']),layer['param']['arg']['keep'])
        self.layers.insert(max(0, len(self.layers)), layer)
        self.args[self.layers[-1]['param']['arg']['keep']] = dropout_keep
        self.args[self.layers[-1]['param']['arg']['steps']] = self.max_steps
        self.is_recurrent = True
        return self

    def __initialize(self, cost_function='MSE'):
        if not self.__is_init:
            self.y = tf.placeholder(tf.float32, [1, 1, self.layers[-1]['n']], name='y')

            if cost_function == 'cross_entropy':
                self.cost_function = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.y,(-1,1)),
                                                            logits=tf.reshape(self.layers[-1]['h'],(-1,1))))

            elif cost_function == 'L2':
                self.cost_function = tf.nn.l2_loss(self.layers[-1]['h'] - self.y)
            else:
                self.cost_function = tf.reduce_mean(tf.squared_difference(self.layers[-1]['h'], self.y))

            correct_prediction = tf.equal(tf.reshape(self.y,(-1,1)), tf.round(tf.reshape(tf.nn.sigmoid(self.layers[-1]['h']),(-1,1))))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.train_step = tf.train.AdamOptimizer(self.step_size).minimize(self.cost_function)
            tf.global_variables_initializer().run()

            self.__is_init = True


    def train(self, x, y, holdout=None, step=0.1, epochs=50, batch=10, cost_method='MSE'):
        print("{:=<50}".format(''))
        print("{:^50}".format("Training Network"))
        print("{:=<50}".format(''))
        structure = "{}n".format(self.layers[0]['n'])
        for i in range(1,len(self.layers)):
            structure += " -> {}n".format(self.layers[i]['n'])
        print("-{} layers: {}".format(len(self.layers),structure))
        print("-{} epochs".format(epochs))
        print("-step size = {}".format(step))
        print("-batch size = {}".format(batch))
        print("{:=<50}".format(''))
        print("{:<10}{:^10}{:^10}{:>10}".format("Epoch","Cost","Acc","Time"))
        print("{:=<50}".format(''))

        h_x = None
        h_y = None
        if holdout is not None:
            keep = np.arange(np.alen(x))
            keep = np.delete(keep, np.searchsorted(keep,holdout))

            h_x = x[holdout]
            h_y = y[holdout]
            x = x[keep]
            y = y[keep]


        self.step_size = step
        self.batch_size = batch
        self.training_epochs = epochs

        self.__initialize(cost_method)

        self.args[self.layers[0]['param']['arg']['mean']] = np.mean(np.hstack([i.ravel() for i in x])
                                                                    .reshape((-1,x[0].shape[1])),
                                                                    axis=0).reshape((1,1,-1))
        self.args[self.layers[0]['param']['arg']['stdev']] = np.std(np.hstack([i.ravel() for i in x])
                                                                    .reshape((-1,x[0].shape[1])),
                                                                    axis=0).reshape((1,1,-1))
        h_cost = []
        train_start = time.time()
        for e in range(epochs):
            epoch_start = time.time()

            cost = []

            acc = []
            for i in range(0, x.shape[0], batch):
                s = range(i, min(x.shape[0], i+batch))

                batch_x = np.array(x[s][0]).tolist()

                self.args[self.layers[1]['param']['arg']['steps']] = len(x[s])

                x_pad = np.zeros(self.layers[0]['n']).tolist()

                while len(batch_x) < self.max_steps:
                    batch_x.append(x_pad)

                batch_x = np.array(batch_x).reshape((1,-1,self.layers[0]['n']))

                self.args[self.layers[0]['z']] = batch_x
                self.args[self.y] = y[s].reshape(1, 1, self.layers[-1]['n'])
                self.train_step.run(feed_dict=self.args)

                cost.append(self.getCost(x[s], y[s], True))
                acc.append(self.getAccuracy(x[s], y[s], True))

            if holdout is not None:
                c = self.getCost(h_x, h_y, True)
                # print(c)
                h_cost.append(c)
                # np.append(h_cost, self.getCost(h_x, h_y, True))
                if np.alen(h_cost) >= 11:
                    prev = np.mean(h_cost[-11:-1])
                    if prev < np.mean(h_cost[-10:]):
                        print("{:<10}{:^10.4f}{:^10.4f}{:>9.1f}s {}".format("Epoch " + str(e + 1), np.mean(cost),
                                                                            np.mean(acc),
                                                                            time.time() - epoch_start,
                                                                            "-- ({:<.4f})".format(np.mean(h_cost[-10:])
                                                                                                  if e >= 10 and holdout is not None
                                                                                                  else h_cost[-1])))
                        break

            print("{:<10}{:^10.4f}{:^10.4f}{:>9.1f}s {}".format("Epoch "+str(e+1), np.mean(cost), np.mean(acc),
                                                                time.time()-epoch_start,
                                                                "-- ({:<.4f})".format(np.mean(h_cost[-10:])
                                                                if e >= 10 and holdout is not None
                                                                else h_cost[-1])))

        print("{:=<50}".format(''))
        print("Total Time: {:<.1f}s".format(time.time()-train_start))

    def predict(self, x, layer=-1):
        arg = dict(self.args)
        for i in arg:
            if 'keep' in i.name:
                arg[i] = 1
        del arg[self.y]
        p = []

        for s in range(len(x)):
            batch_x = np.array(x[s]).tolist()

            arg[self.layers[1]['param']['arg']['steps']] = len(x[s])

            x_pad = np.zeros(self.layers[0]['n']).tolist()

            while len(batch_x) < self.max_steps:
                batch_x.append(x_pad)

            batch_x = np.array(batch_x).reshape((1, -1, self.layers[0]['n']))

            arg[self.layers[0]['z']] = batch_x
            p.append(tf.nn.sigmoid(self.layers[layer]['h']).eval(feed_dict=arg))
        return np.array(p)

    def getAccuracy(self, x, y, test=True):
        arg = dict(self.args)
        if test:
            for i in arg:
                if 'keep' in i.name:
                    arg[i] = 1
        acc = []
        for s in range(len(x)):
            batch_x = np.array(x[s]).tolist()

            arg[self.layers[1]['param']['arg']['steps']] = len(x[s])

            x_pad = np.zeros(self.layers[0]['n']).tolist()

            while len(batch_x) < self.max_steps:
                batch_x.append(x_pad)

            batch_x = np.array(batch_x).reshape((1, -1, self.layers[0]['n']))

            arg[self.layers[0]['z']] = batch_x
            arg[self.y] = y[s].reshape(1, 1, -1)
            acc.append(self.accuracy.eval(feed_dict=arg))
        return np.mean(acc)

    def getCost(self, x, y, test=True):
        arg = dict(self.args)
        if test:
            for i in arg:
                if 'keep' in i.name:
                    arg[i] = 1
        cost = []
        for s in range(len(x)):

            batch_x = np.array(x[s]).tolist()

            arg[self.layers[1]['param']['arg']['steps']] = len(x[s])

            x_pad = np.zeros(self.layers[0]['n']).tolist()

            while len(batch_x) < self.max_steps:
                batch_x.append(x_pad)

            batch_x = np.array(batch_x).reshape((1, -1, self.layers[0]['n']))

            arg[self.layers[0]['z']] = batch_x
            arg[self.y] = y[s].reshape(1, 1, -1)
            cost.append(self.cost_function.eval(feed_dict=arg))
        return np.mean(cost)



def loadData(x_file, y_file, x_stride=15, n=None):


    x = []
    y = []

    with open(x_file, 'r') as xf, open(y_file, 'r') as yf:
        seq = []
        xf_lines = xf.readlines()
        yf_lines = yf.readlines()
        for line in range(len(xf_lines)):
            if line == 0:
                continue  # skip header
            elif int((line/float(x_stride))-1) == n:
                break

            csv = np.array(xf_lines[line].strip().split(','))
            for j in range(10):
                try:
                    seq.append(np.array([  float(csv[j+int(0*10)])  # correctness
                                          # ,float(csv[j+int(1*10)])  # hints
                                          # ,float(csv[j+int(2*10)])  # attempts
                                          # ,float(csv[j+int(3*10)])  # time on problem
                                          # ,float(csv[j+int(4*10)])  # first response time
                                         ]))
                except ValueError:
                    pass
            if line % int(x_stride) == 0:
                x.append(np.array(seq))
                one_hot = np.zeros((2),dtype=float)
                one_hot[int(yf_lines[int((line/float(x_stride)))].strip().split(',')[0])] = 1
                y.append([int(yf_lines[int((line/float(x_stride)))].strip().split(',')[0])])
                seq = []

    return np.array(x), np.array(y)


def Aprime(actual, predicted):
    assert len(actual) == len(predicted)

    score = [[],[]]

    for i in range(0,len(actual)):
        score[int(actual[i])].append(predicted[i])

    sum = 0.0
    for p in score[1]:
        for n in score[0]:
            if p > n:
                sum += 1
            elif p == n:
                sum += .5
            else:
                sum += 0

    return sum/(float(len(score[0]))*len(score[1]))


if __name__ == "__main__":

    # x, y = loadData('Dataset/training_images.csv', 'Dataset/training_labels.csv')
    #
    # x_t, y_t = loadData('Dataset/workshop/testing_images.csv', 'Dataset/workshop/testing_labels.csv')

    from numpy import genfromtxt

    np.random.seed(0)
    tf.set_random_seed(0)

    training_temp = genfromtxt('Dataset/workshop/training_images.csv', dtype=int, delimiter=',', names=True)
    testing_temp = genfromtxt('Dataset/workshop/testing_images.csv', dtype=int, delimiter=',', names=True)
    train_labels_temp = genfromtxt('Dataset/workshop/training_labels.csv', dtype=int, delimiter=',', names=True)
    test_labels_temp = genfromtxt('Dataset/workshop/testing_labels.csv', dtype=int, delimiter=',', names=True)

    training = training_temp.view(np.int).reshape(len(training_temp), -1)
    testing = testing_temp.view(np.int).reshape(len(testing_temp), -1)
    y = train_labels_temp.view(np.int).reshape((-1, 1))
    y_t = test_labels_temp.view(np.int).reshape((-1, 1))
    # print test_labels
    batch = 15

    x = []
    x_t = []

    nchannels = 5

    for i in range(0, training.shape[0], batch):
        imgSize = range(i, min(training.shape[0], i + batch))
        ch = []
        for j in range(nchannels):
            ch.append(np.array(training[imgSize, (j*10):((j*10)+10)]).reshape(( -1, 1)))
        ch = np.array(ch).T[0]
        x.append(ch[np.where(ch[:, 0] >= 0),:].reshape((-1,nchannels)))

    for i in range(0, testing.shape[0], batch):
        imgSize = range(i, min(testing.shape[0], i + batch))
        ch = []
        for j in range(nchannels):
            ch.append(np.array(testing[imgSize, (j * 10):((j * 10) + 10)]).reshape((-1, 1)))
        ch = np.array(ch).T[0]
        x_t.append(ch[np.where(ch[:, 0] >= 0), :].reshape((-1, nchannels)))

    x = np.array(x)
    x_t = np.array(x_t)
    y = np.array(y, dtype=float)
    y_t = np.array(y_t, dtype=float)

    net = Network() \
        .add_input_layer(x[0].shape[-1], True) \
        .add_LSTM_layer(10, use_last=True,activation=tf.nn.relu,dropout_keep=.5) \
        .add_dense_layer(1)

    net.train(x, y, holdout=np.arange(450), epochs=200, step=1e-4, batch=1, cost_method='cross_entropy')

    print("Test Accuracy: ", net.getAccuracy(x_t,y_t))
    print("Test Cost: ", net.getCost(x_t, y_t))
    pred = net.predict(x_t).reshape((-1,1))
    print("Test AUC: ", Aprime(y_t.reshape(-1,1),pred))



