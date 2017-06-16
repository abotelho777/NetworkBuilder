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

        self.session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))  # use CPU for now

    def add_input_layer(self, n):
        layer = dict()
        layer['n'] = n
        layer['z'] = tf.placeholder(tf.float32, [1, self.max_steps, n], name='x')
        layer['param'] = {'w': None, 'b': None, 'type': 'input',
                          'arg': {'mean': tf.placeholder(tf.float32, name='input_mean'),
                                  'stdev': tf.placeholder(tf.float32, name='input_stdev')}}
        layer['a'] = tf.identity
        layer['h'] = layer['a']((layer['z']-layer['param']['arg']['mean']) /
                                tf.maximum(layer['param']['arg']['stdev'], tf.constant(1e-12, dtype=tf.float32)))

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

        # self.layers[-1]['h'].set_shape([-1, self.max_steps, self.layers[0]['n']])

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
                # self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.layers[-1]['h']),
                #                                                    reduction_indices=[1]))
            elif cost_function == 'L2':
                self.cost_function = tf.nn.l2_loss(self.layers[-1]['h'] - self.y)
            else:
                self.cost_function = tf.reduce_mean(tf.squared_difference(self.layers[-1]['h'], self.y))


            correct_prediction = tf.equal(tf.reshape(self.y,(-1,1)), tf.round(tf.reshape(tf.nn.sigmoid(self.layers[-1]['h']),(-1,1))))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.auc = tf.metrics.auc(tf.argmax(self.y, 2),tf.argmax(self.layers[-1]['h'], 1))


            self.train_step = tf.train.AdagradOptimizer(self.step_size).minimize(self.cost_function)
            tf.global_variables_initializer().run()


            self.__is_init = True

    def train_sequence(self, x, y, step=0.1, epochs=50, batch=10, cost_method='MSE'):
        if not self.is_recurrent:
            return self.train(x, y, step, epochs, batch, cost_method)

        if len(x.shape) != 3:
            raise ValueError('Invalid input dimensionality. Input must be presented as [sequence, time step, covariates].')

    def train(self, x, y, step=0.1, epochs=50, batch=10, cost_method='MSE'):

        # if self.is_recurrent:
        #     return self.train_sequence(x,y,step,epochs,batch,cost_method)

        # if len(x.shape) != 2:
        #     raise ValueError('Invalid input dimensionality. Input must be presented as [sample, covariates].')

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

        train_start = time.time()
        for e in range(epochs):
            epoch_start = time.time()

            cost = []
            acc = []
            auc = []
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
                # print(self.predict(x[s]), y[s], self.getAccuracy(x[s], y[s], True))

                cost.append(self.getCost(x[s], y[s], True))
                acc.append(self.getAccuracy(x[s], y[s], True))
                # auc.append(self.getAUC(x[s], y[s], False))

            print("{:<10}{:^10.4f}{:^10.4f}{:>9.1f}s".format("Epoch "+str(e+1), np.mean(cost), np.mean(acc),
                                                    time.time()-epoch_start))

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

    def getAUC(self, x, y, test=True):

        return tf.metrics.auc(np.argmax(y,1),np.argmax(self.predict(x),2).flatten())


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

            csv = xf_lines[line].strip().split(',')
            for j in range(len(csv)):
                try:
                    seq.append(np.array([float(csv[j])]))
                except ValueError:
                    pass
            if line % int(x_stride) == 0:
                x.append(np.array(seq))
                one_hot = np.zeros((2),dtype=float)
                one_hot[int(yf_lines[int((line/float(x_stride)))].strip().split(',')[0])] = 1
                y.append([int(yf_lines[int((line/float(x_stride)))].strip().split(',')[0])])
                seq = []

    return np.array(x), np.array(y)


def loadCSV(filename, max_rows=None):
    csvarr = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            # split out each comma-separated value
            name = line.strip().split(',')
            for j in range(0,len(name)):
                # try converting to a number, if not, leave it
                try:
                    name[j] = float(name[j])
                except ValueError:
                    # do nothing constructive
                    name[j] = name[j]
            csvarr.append(name)
            if max_rows is not None:
                if len(csvarr) >= max_rows:
                    break
    return csvarr


def writetoCSV(ar,filename,headers=[]):
    # ar = np.array(transpose(ar))
    ar = np.array(ar)
    assert len(ar.shape) <= 2

    with open(filename + '.csv', 'w') as f:
        if len(headers)!=0:
            for i in range(0,len(headers)-1):
                f.write(str(headers[i]) + ',')
            f.write(str(headers[len(headers)-1])+'\n')
        for i in range(0,len(ar)):
            if (len(ar.shape) == 2):
                for j in range(0,len(ar[i])-1):
                    f.write(str(ar[i][j]) + ',')
                f.write(str(ar[i][len(ar[i])-1]) + '\n')
            else:
                f.write(str(ar[i]) + '\n')
    f.close()


def loadCSVwithHeaders(filename, max_rows=None):
    if max_rows is not None:
        max_rows += 1
    data = loadCSV(filename,max_rows)
    headers = np.array(data[0])
    # data = np.array(convert_to_floats(data))
    data = np.delete(data, 0, 0)
    return data,headers


def readHeadersCSV(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            # split out each comma-separated value
            return line.strip().split(',')
    return []



def reshape_sequence(table, pivot, labels=None, columns=None, order=None, sequence_label=False):
    if columns is None:
        columns = range(table.shape[-1])
    col = np.array(columns)

    p = np.array(np.unique(table[:,pivot]))
    seq = {'key': [], 'x': [], 'y': []}
    for i in p:
        match = np.where(table[:,pivot] == i)

        seq['key'].append(i)

        seq['x'].append(np.array(table[match, col.reshape((-1, 1))],dtype=np.float32).T[np.argsort(table[match, order])]\
            .reshape((-1, len(col))))

        lab = None
        if labels is not None:
            y = np.array(table[match, np.array(labels).reshape((-1, 1))],dtype=np.float32).T[np.argsort(table[match, order])]\
                .reshape((-1, len(labels)))
            lab = y if not sequence_label else y[-1]
        seq['y'].append(lab)
    seq['key'] = np.array(seq['key'])
    seq['x'] = np.array(seq['x'])
    seq['y'] = np.array(seq['y'])
    return seq

def Aprime(actual, predicted):
    assert len(actual) == len(predicted)

    # print(actual[0:15])
    #
    # print(predicted[0:15])

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

    # data = np.array( [[1, 2, 3, 4, 0, 1],
    #                [1, 2, 5, 6, 1, 0],
    #                [2, 2, 7, 6, 1, 0],
    #                [2, 1, 8, 9, 0, 1],
    #                [3, 1, 2, 3, 1, 0]] )
    #
    # data, headers = loadCSVwithHeaders('Dataset/wsRNN.csv', 10000)
    #
    # for i in range(0, len(headers)):
    #     print('{:>2}:  {:<18} {:<12}'.format(str(i), headers[i], data[0][i]))
    #
    # sequences = reshape_sequence(data, 2, [12], [6], 0, True)
    #
    # x = sequences['x']
    #
    #
    # y = sequences['y']
    #

    x, y = loadData('Dataset/training_images.csv', 'Dataset/training_labels.csv')

    x_t, y_t = loadData('Dataset/testing_images.csv', 'Dataset/testing_labels.csv')

    net = Network() \
        .add_input_layer(x[0].shape[-1]) \
        .add_LSTM_layer(10, use_last=True,activation=tf.nn.relu,dropout_keep=.5) \
        .add_dense_layer(1)

    net.train(x, y, epochs=20, step=.1, batch=1, cost_method='cross_entropy')

    print("Test Accuracy: ", net.getAccuracy(x_t,y_t))
    print("Test Cost: ", net.getCost(x_t, y_t))
    pred = net.predict(x_t).reshape((-1,1))
    # print("Test AUC: ", tf.metrics.auc(y_t.reshape(-1,1),pred))
    print("Test AUC: ", Aprime(y_t.reshape(-1,1),pred))
    # print("MC AUC: ", Aprime(y_t.reshape(-1,1), np.ones(pred.shape)))
    #
    # print("Test AUC: ", net.getAUC(x_t, y_t))
    # print(net.predict(x[0:10, :], layer=-3))


    #
    # training_img = np.load('Dataset/mnist_train_images.npy')
    # training_lab = np.load('Dataset/mnist_train_labels.npy')
    # testing_img = np.load('Dataset/mnist_test_images.npy')
    # testing_lab = np.load('Dataset/mnist_test_labels.npy')
    # validation_img = np.load('Dataset/mnist_validation_images.npy')
    # validation_lab = np.load('Dataset/mnist_validation_labels.npy')
    #
    # print('Training Samples: {:>10}'.format(len(training_img)))
    # print('Testing Samples:  {:>10}'.format(len(testing_img)))
    # print('Sample Shape:     {:>10}'.format(str(training_img.shape)))
    # print('Label Shape:      {:>10}'.format(str(training_lab.shape)))
    # print('\n')
    # print_label_distribution(training_lab)
    # print('\n')
    #
    # np.random.seed(0)
    # tf.set_random_seed(0)
    #
    # # Build Network Structure
    # # net = Network()\
    # #     .add_input_layer(training_img.shape[1])\
    # #     .add_dropout_layer(h, 0.5, tf.nn.relu)\
    # #     .add_dense_layer(training_lab.shape[1], tf.nn.softmax)
    # #
    # # Train network
    # # net.train(training_img, training_lab, epochs=e, step=s, batch=b, cost_method='cross_entropy')
    #
    # net = Network() \
    #     .add_input_layer(training_img.shape[1]) \
    #     .add_dropout_layer(int(training_img.shape[1]/2), 0.5, tf.nn.relu) \
    #     .add_dropout_layer(int(training_img.shape[1]/4), 0.5, tf.identity) \
    #     .add_inverse_layer(2, activation=tf.identity) \
    #     .add_inverse_layer(1, activation=tf.identity)
    #
    # net.train(training_img, (training_img - np.mean(training_img, axis=0)) /
    #           np.maximum(1e-12, np.std(training_img, axis=0)),
    #           epochs=20, step=1e-1, batch=64, cost_method='L2')
    #
    # print(net.predict(training_img[0:10, :], layer=-3))






