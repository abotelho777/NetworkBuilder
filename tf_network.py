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

        self.args = dict()

        self.outputs = None

        self.deepest_hidden_layer = None

        self.recurrent = False
        self.use_last = False
        self.deepest_recurrent_layer = None

        self.__max_time_backprop = 3

        # self.session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))  # use CPU
        self.session = tf.InteractiveSession()  # use GPU

    def add_input_layer(self, n):
        layer = dict()
        layer['n'] = n
        layer['z'] = tf.placeholder(tf.float32, [None, n], name='x')
        layer['param'] = {'w': None, 'b': None, 'type': 'input',
                          'arg': {'mean': tf.placeholder(tf.float32, name='input_mean'),
                                  'stdev': tf.placeholder(tf.float32, name='input_stdev')}}
        layer['a'] = tf.identity
        layer['h'] = layer['a']((layer['z']-layer['param']['arg']['mean']) /
                                tf.maximum(layer['param']['arg']['stdev'], tf.constant(1e-12, dtype=tf.float32)))

        self.layers.insert(max(0, len(self.layers)), layer)
        self.args[self.layers[-1]['param']['arg']['mean']] = 0
        self.args[self.layers[-1]['param']['arg']['stdev']] = 1
        self.deepest_hidden_layer = self.layers[-1]
        return self

    def add_dense_layer(self, n, activation=tf.identity):
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'dense', 'arg': None}
        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32,name='Layer'+str(len(self.layers))+'_W')
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]),name='Layer'+str(len(self.layers))+'_B')
        layer['z'] = tf.matmul(self.layers[-1]['h'], layer['param']['w']) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
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
        layer['z'] = tf.matmul(self.layers[-1]['h'], tf.transpose(inv['param']['w'][:(-inv['n']),:])) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        return self

    def __add_gate(self, n, feeding_n, matrix_merge, activation=tf.identity):
        gate = dict()
        gate['n'] = n
        gate['param'] = {'w': None, 'b': None, 'type': 'gate',
                          'arg': None}

        gate['param']['w'] = tf.Variable(tf.truncated_normal((feeding_n, gate['n']),
                                                             stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                         dtype=tf.float32)
        gate['param']['b'] = tf.Variable(tf.zeros([gate['n']]))

        concat = tf.concat([tf.reshape(self.layers[-1]['h'],[-1, self.layers[-1]['n']]), matrix_merge], 1)
        gate['z'] = tf.matmul(concat, gate['param']['w']) + gate['param']['b']

        gate['a'] = activation
        gate['h'] = gate['a'](gate['z'])
        return gate

    def add_lstm_layer(self, n, use_last=False, peepholes=False, activation=tf.identity):
        self.recurrent = True
        self.use_last = use_last
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'recurrent',
                          'arg': {'hsubt': None, 'cell': None}}

        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n']+layer['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32)
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]), name='cell_state')

        layer['param']['arg']['hsubt'] = tf.placeholder(tf.float32, [None, layer['n']], name='cell_hsubt')
        layer['param']['arg']['cell'] = tf.Variable(tf.zeros([layer['n']]))

        concat = tf.concat([tf.reshape(self.layers[-1]['h'], [-1, self.layers[-1]['n']]),
                            layer['param']['arg']['hsubt']], 1)

        cell_prime = tf.tanh(tf.matmul(concat, layer['param']['w']) + layer['param']['b'])

        feeding_n = self.layers[-1]['n']+n

        if peepholes:
            feeding_n += n
            p_concat = tf.concat([layer['param']['arg']['hsubt'],
                                  tf.reshape(tf.tile(layer['param']['arg']['cell'],
                                                     [tf.shape(layer['param']['arg']['hsubt'])[0]]),
                                             [-1, layer['n']])], 1)
        else:
            p_concat = layer['param']['arg']['hsubt']

        forget_g = self.__add_gate(n, feeding_n, p_concat, tf.sigmoid)
        input_g = self.__add_gate(n, feeding_n, p_concat, tf.sigmoid)

        layer['param']['arg']['cell'] = (layer['param']['arg']['cell'] * forget_g['h']) + \
                                        (cell_prime * input_g['h'])

        if peepholes:
            pr_concat = tf.concat([layer['param']['arg']['hsubt'], layer['param']['arg']['cell']], 1)
        else:
            pr_concat = layer['param']['arg']['hsubt']

        output_g = self.__add_gate(n, feeding_n, pr_concat, tf.sigmoid)

        layer['z'] = (output_g['h'] * tf.tanh(layer['param']['arg']['cell']))
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.deepest_recurrent_layer = self.layers[-1]
        return self

    def add_gru_layer(self, n, use_last=False, activation=tf.identity):
        self.recurrent = True
        self.use_last = use_last
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'recurrent',
                          'arg': {'hsubt': None, 'cell': None}}

        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n']+layer['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32)
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]))

        layer['param']['arg']['hsubt'] = tf.placeholder(tf.float32, [None, layer['n']], name='cell_hsubt')

        update_g = self.__add_gate(n, self.layers[-1]['n']+n , layer['param']['arg']['hsubt'], activation=tf.sigmoid)
        reset_g = self.__add_gate(n, self.layers[-1]['n']+n, layer['param']['arg']['hsubt'], activation=tf.sigmoid)

        concat = tf.concat([tf.reshape(self.layers[-1]['h'], [-1, self.layers[-1]['n']]),
                            reset_g['h'] * layer['param']['arg']['hsubt']], 1)

        cell_prime = tf.tanh(tf.matmul(concat, layer['param']['w']) + layer['param']['b'])

        layer['z'] = (1-update_g['h'])*layer['param']['arg']['hsubt'] + update_g['h'] * cell_prime
        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.deepest_recurrent_layer = self.layers[-1]
        return self

    def add_rnn_layer(self, n, use_last=False, activation=tf.identity):
        self.recurrent = True
        self.use_last = use_last
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'recurrent',
                          'arg': {'init': None, 'hsubt': None, 'cell': None}}

        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n'] + layer['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32,name='Layer'+str(len(self.layers))+'_W')
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]), name='Layer'+str(len(self.layers))+'_B')

        layer['param']['arg']['hsubt'] = tf.placeholder(tf.float32, [None, layer['n']], name='cell_hsubt')

        concat = tf.concat([tf.reshape(self.layers[-1]['h'],[-1,self.layers[-1]['n']]),
                            layer['param']['arg']['hsubt']], 1)

        layer['z'] = tf.tanh(tf.matmul(concat, layer['param']['w']) + layer['param']['b'])

        layer['a'] = activation
        layer['h'] = layer['a'](layer['z'])
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.deepest_recurrent_layer = self.layers[-1]
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
        self.deepest_hidden_layer = self.layers[-1]
        return self

    def set_max_backprop_timesteps(self, timesteps):
        self.__max_time_backprop = timesteps

    def __initialize(self, cost_function='MSE'):
        if not self.__is_init:
            self.y = tf.placeholder(tf.float32, [None, self.layers[-1]['n']], name='y')

            if cost_function == 'cross_entropy':
                self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.layers[-1]['h']),
                                                                   reduction_indices=[1]))
            elif cost_function == 'L2':
                self.cost_function = tf.nn.l2_loss(self.layers[-1]['h'] - self.y)
            elif cost_function == 'rmse':
                self.cost_function = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.layers[-1]['h'], self.y)))
            else:
                self.cost_function = tf.reduce_mean(tf.squared_difference(self.layers[-1]['h'], self.y))

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.layers[-1]['h'], 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.update = tf.train.AdagradOptimizer(self.step_size, name='update')
            self.minimize_cost = self.update.minimize(self.cost_function)

            self.var_grads = self.update.compute_gradients(self.cost_function, tf.trainable_variables())

            self.gradients = [(tf.placeholder(tf.float32,g.get_shape(),
                                              name='g_'+v.name.replace(':','_')),v) for g,v in self.var_grads]
            self.update_weights = self.update.apply_gradients(self.gradients)

            self.outputs = [self.layers[-1]['h']]

            tf.global_variables_initializer().run()

            tf.get_default_graph().finalize()

            self.__is_init = True

    def __backprop_through_time(self, x, y, s):
        batch_cost = []
        sequence_input = [[] for _ in range(len(x[s]))]
        sequence_states = []
        sequence_labels = [[] for _ in range(len(x[s]))]
        recurrent_layer = 0
        self.outputs = [self.layers[0]['z']]
        for j in range(len(self.layers)):
            if self.layers[j]['param']['type'] == 'recurrent':
                recurrent_layer = j
                self.outputs.insert(max(0, len(self.outputs) - 1), self.layers[j]['z'])
                self.args[self.layers[j]['param']['arg']['hsubt']] = \
                    np.ones((len(s), self.layers[j]['n'])) * 0.5

        for b in range(len(self.outputs[:-1])):
            sequence_states.append([[] for _ in range(len(x[s]))])

        valid = np.argwhere(np.array([len(k) for k in x[s]]) > 0)
        invalid = np.argwhere(np.array([len(k) for k in x[s]]) <= 0)

        series_batch = x[s][valid]
        series_label = None
        if not self.use_last:
            series_label = y[s][valid]
        n_timestep = max([len(k) for k in x[s]])

        for j in range(n_timestep):
            timestep = []
            timestep_label = []
            for k in range(len(series_batch)):

                timestep.append(np.concatenate(series_batch[k][0]).reshape((-1, self.layers[0]['n']))[j])
                if not self.use_last:
                    timestep_label.append(np.concatenate(
                        series_label[k][0]).reshape((-1, self.layers[-1]['n']))[j])
            timestep = np.array(timestep)

            self.args[self.layers[0]['z']] = np.array(timestep)
            self.args[self.y] = np.array(timestep_label)

            timestep_output = self.session.run(self.outputs, feed_dict=self.args)

            for k in range(len(timestep_output[-1])):
                sequence_input[valid.ravel()[k]].append(timestep_output[-1][k])
                sequence_labels[valid.ravel()[k]].append(timestep_label[k])
                for b in range(len(timestep_output[:-1])):
                    sequence_states[b][valid.ravel()[k]].append(timestep_output[b][k])

            for k in invalid.ravel():
                sequence_input[k].append(np.array([None for _ in range(self.layers[-1]['n'])]))
                sequence_labels[k].append(np.array([None for _ in range(self.layers[-1]['n'])]))
                for b in range(len(timestep_output[:-1])):
                    sequence_states[b][k].append(np.array([None]))

            valid = np.argwhere(np.array([len(k) for k in x[s]]) > j + 1)
            invalid = np.argwhere(np.array([len(k) for k in x[s]]) <= j + 1)
            batch_valid = np.argwhere(np.array([len(k[0]) for k in series_batch]) > j + 1).ravel()

            if len(valid) == 0:
                break

            series_batch = x[s][valid]
            series_label = None
            if not self.use_last:
                series_label = y[s][valid]

            m = 0
            for k in range(len(self.layers)):
                if self.layers[k]['param']['type'] == 'recurrent':
                    self.args[self.layers[k]['param']['arg']['hsubt']] = timestep_output[m][batch_valid] \
                        .reshape((-1, self.layers[k]['n']))
                    m += 1

        sequence_input = np.array(sequence_input)
        sequence_labels = np.array(sequence_labels)
        sequence_states = np.array(sequence_states)

        gradients = None
        for j in range(n_timestep - 1, -1, -1):
            labeled = np.argwhere([all(m is not None for m in k) for k in sequence_labels[:, j]]).ravel()
            if len(labeled) == 0:
                continue

            grad_bptt = None
            for m in range(j, max(min(j - self.__max_time_backprop, j)-1, -1), -1):
                b = 0
                for k in range(len(self.layers)):
                    if self.layers[k]['param']['type'] == 'recurrent':
                        self.args[self.layers[k]['param']['arg']['hsubt']] = \
                            np.concatenate(np.array(np.array(sequence_states[b])[labeled])[:, m]) \
                                .reshape((-1, self.layers[k]['n']))
                        b += 1

                self.args[self.layers[0]['z']] = np.array(sequence_input[:, m])[labeled]
                self.args[self.y] = np.array(sequence_labels[:, j])[labeled]

                batch_cost.append(self.getCost(np.array(sequence_input[:, m])[labeled],
                                         np.array(sequence_labels[:, j])[labeled],
                                         False))

                grads = np.array(self.session.run([self.var_grads], feed_dict=self.args))[0] / len(labeled)
                if grad_bptt is None:
                    grad_bptt = np.array(grads[:, 0])
                else:
                    grad_bptt[:((recurrent_layer * 2) + 1)] += \
                        np.array(np.array(grads[:, 0]))[:((recurrent_layer * 2) + 1)]
            if gradients is None and grad_bptt is not None:
                gradients = grad_bptt
            elif grad_bptt is not None:
                gradients += grad_bptt

            gradients = np.array(gradients) / n_timestep
            feed = {}
            for k in range(len(self.var_grads)):
                feed[self.gradients[k][0]] = gradients[k]
            self.session.run(self.update_weights, feed_dict=feed)
            gradients = None
        return batch_cost

    def train(self, x, y, step=0.1, max_epochs=100, threshold=0.01, batch=10, cost_method='MSE'):

        print("{:=<40}".format(''))
        print("{:^40}".format("Training Network"))
        print("{:=<40}".format(''))
        structure = "{}n".format(self.layers[0]['n'])
        for i in range(1, len(self.layers)):
            structure += " -> {}n".format(self.layers[i]['n'])
        print("-{} layers: {}".format(len(self.layers), structure))
        print("-{} epochs".format(max_epochs))
        print("-step size = {}".format(step))
        print("-batch size = {}".format(batch))
        print("{:=<40}".format(''))
        print("{:<10}{:^10}{:>10}".format("Epoch", "Cost", "Time"))
        print("{:=<40}".format(''))

        self.step_size = step
        self.batch_size = batch
        self.training_epochs = max_epochs

        self.__initialize(cost_method)

        self.args[self.layers[0]['param']['arg']['mean']] = np.mean(np.hstack([np.array(i).ravel() for i in x])
                                                                    .reshape((-1, np.array(x[0]).shape[1])),
                                                                    axis=0).reshape((1, 1, -1))
        self.args[self.layers[0]['param']['arg']['stdev']] = np.std(np.hstack([np.array(i).ravel() for i in x])
                                                                    .reshape((-1, np.array(x[0]).shape[1])),
                                                                    axis=0).reshape((1, 1, -1))

        train_start = time.time()

        e = 1
        while True:
            epoch_start = time.time()

            v = list(range(x.shape[0]))
            np.random.shuffle(v)
            x = x[v]
            y = y[v]

            cost = []
            for i in range(0, x.shape[0], batch):
                s = range(i, min(x.shape[0], i + batch))
                if len(s) < batch:
                    continue

                if self.recurrent:
                    batch_cost = self.__backprop_through_time(x, y, s)
                    for j in batch_cost:
                        cost.append(j)
                else:
                    self.args[self.layers[0]['z']] = x[s]
                    self.args[self.y] = y[s]
                    self.minimize_cost(self.cost_function).run(feed_dict=self.args)
                    cost.append(self.getCost(x[s], y[s], False))

            if e > 1:
                mean_last_ten = np.mean(cost[-10:])
            else:
                mean_last_ten = 0

            print("{:<10}{:^10.4f}{:>9.1f}s".format("Epoch " + str(e), np.mean(cost),
                                                    time.time() - epoch_start))

            if (0.0001 < abs(np.mean(cost[-10:]) - mean_last_ten) < threshold) or e >= max_epochs:
                break;
            e += 1

        print("{:=<40}".format(''))
        print("Total Time: {:<.1f}s".format(time.time() - train_start))

    def predict(self, x, layer=-1):
        arg = dict(self.args)
        for i in arg:
            if 'keep' in i.name:
                arg[i] = 1
        del arg[self.y]

        if self.recurrent:
            pred = [[] for j in range(len(x))]

            for j in range(len(self.layers)):
                if self.layers[j]['param']['type'] == 'recurrent':
                    arg[self.layers[j]['param']['arg']['hsubt']] = np.ones((len(x), self.layers[j]['n'])) * 0.5

            valid = np.argwhere(np.array([len(k) for k in x]) > 0)
            series_batch = x[valid]

            n_timestep = max([len(k) for k in x])

            for j in range(n_timestep):
                timestep = []
                for k in range(len(series_batch)):
                    timestep.append(series_batch[k][0].reshape((-1, self.layers[0]['n']))[j])

                timestep = np.array(timestep)
                arg[self.layers[0]['z']] = timestep
                pred_step = self.layers[layer]['h'].eval(feed_dict=arg)

                for k in range(len(pred_step)):
                    pred[valid.ravel()[k]].append(pred_step[k])

                valid = np.argwhere(np.array([len(k) for k in x]) > j+1)
                batch_valid = np.argwhere(np.array([len(k[0]) for k in series_batch]) > j + 1).ravel()

                if len(valid) == 0:
                    break

                series_batch = x[valid]

                for k in range(len(self.layers)):
                    if self.layers[k]['param']['type'] == 'recurrent':
                        arg[self.layers[k]['param']['arg']['hsubt']] = \
                            np.array(self.layers[k]['h'].eval(feed_dict=arg)[batch_valid])\
                            .reshape((-1, self.layers[k]['n']))
            return pred
        else:
            arg[self.layers[0]['z']] = x
            return self.layers[layer]['h'].eval(feed_dict=arg)

    def getAccuracy(self, x, y, test=True):
        arg = dict(self.args)
        if test:
            for i in arg:
                if 'keep' in i.name:
                    arg[i] = 1
        arg[self.layers[0]['z']] = x
        arg[self.y] = y
        return self.accuracy.eval(feed_dict=arg)

    def getCost(self, x, y, test=True):
        arg = dict(self.args)
        if test:
            for i in arg:
                if 'keep' in i.name:
                    arg[i] = 1
        arg[self.layers[0]['z']] = x
        arg[self.y] = y
        return self.cost_function.eval(feed_dict=arg)


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


def softmax(z):
    t = np.exp(z - z.max())
    return t / np.sum(t, axis=1, keepdims=True)


def print_label_distribution(labels, label_names=None):
    print("\nLabel Distribution:")

    n = np.sum(np.array(labels), axis=0)
    dist = softmax(np.array([n/np.max(n)]))[0]

    if label_names is not None:
        assert len(label_names) == len(dist)
    else:
        label_names = []
        for i in range(0, len(dist)):
            label_names.append("Label_" + str(i))

    for i in range(0, len(dist)):
        print("   " + label_names[i] + ":", "{:<6}".format(int(n[i])), "({0:.0f}%)".format(dist[i]*100))


def flatten_sequence(sequence, include_sample_num=False):
    ar = []
    for i in range(len(sequence)):
        for j in sequence[i]:
            if include_sample_num:
                row = [i]
                for k in j:
                    row.append(k)
                ar.append(row)
            else:
                ar.append(j)

    return np.array(ar)


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


def generate_timeseries_test(samples=1000, max_sequence=50, regular_offset=None, categorical=False):
    data = []
    labels = []

    if max_sequence == 1:
        r = np.ones(samples)
    else:
        r = np.random.randint(1,max_sequence,samples)

    for i in range(samples):
        x = np.random.rand()*10
        seq = []
        lab = []
        for j in range(int(r[i])):
            if regular_offset is None:
                offset = np.random.rand()
            else:
                offset = regular_offset
            # seq.append(np.array([x,offset]))
            seq.append(np.array([np.sin(x)]))
            # lab.append(np.array([np.round(np.sin(x)) % 2, 1-np.round(np.sin(x)) % 2]))

            if categorical:
                lab.append(np.array([np.round(np.sin(x+offset)) % 2, 1 - np.round(np.sin(x+offset)) % 2]))
            else:
                lab.append(np.array([np.sin(x+offset) if np.random.rand() > 0.2 else None]))
                # lab.append(np.array([np.sin(x + offset)]))
            x += offset
        data.append(np.array(seq))
        labels.append(np.array(lab))

    return np.array(data), np.array(labels)


if __name__ == "__main__":
    np.random.seed(1)
    data, labels = generate_timeseries_test(samples=1000, max_sequence=20, regular_offset=0.3, categorical=False)

    np.random.seed(0)
    tf.set_random_seed(0)

    net = Network() \
        .add_input_layer(data[0].shape[-1]) \
        .add_rnn_layer(2, activation=tf.identity) \
        .add_dense_layer(labels[0].shape[-1], activation=tf.identity)

    net.set_max_backprop_timesteps(3)

    net.train(data, labels, max_epochs=50, step=1e-2, batch=100, cost_method='rmse', threshold=0.001)

    pred = net.predict(np.sin(np.array(range(30))*0.3).reshape((1,-1,1)))

    fp = np.array(flatten_sequence(pred,True))
    fp[:, 0] = np.sin(np.array(range(30))*0.3)
    writetoCSV(fp, 'predictions')