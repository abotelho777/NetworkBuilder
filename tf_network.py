import numpy as np
import tensorflow as tf
import time
from evaluationutility import *

class Normalization:
    NONE = 'None'
    Z_SCORE = 'z_score'
    MAX = 'max'


class Cost:
    NONE = 'None'
    MSE = 'MSE'
    L2_NORM = 'L2'
    CROSS_ENTROPY = 'cross_entropy'
    RMSE = 'rmse'



class Network:
    def __init__(self):
        self.layers = []
        self.__is_init = False
        self.step_size = None
        self.batch_size = None

        self.training_epochs = None

        self.args = dict()

        self.normalization = Normalization.NONE
        self.outputs = None

        self.deepest_hidden_layer = None

        self.recurrent = False
        self.use_last = False
        self.deepest_recurrent_layer = None
        self.__max_timesteps = 50

        self.__max_time_backprop = 3

        self.__deepest_hidden_layer_ind = None

        self.graph = tf.get_default_graph()

        # self.session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))  # use CPU
        self.session = tf.InteractiveSession()  # use GPU

    def add_input_layer(self, n, normalization=Normalization.NONE):
        layer = dict()
        layer['n'] = n
        layer['z'] = tf.placeholder(tf.float32, [None, None, n], name='x')
        layer['param'] = {'w': None, 'b': None, 'type': 'input',
                          'arg': {'stat1': tf.placeholder_with_default(tf.zeros([n]), shape=[n],
                                                                       name='input_stat1'),
                                  'stat2': tf.placeholder_with_default(tf.ones([n]), shape=[n],
                                                                       name='input_stat2')}}

        layer['a'] = tf.identity

        self.normalization = normalization
        if normalization == Normalization.Z_SCORE:
            layer['h'] = layer['a']((layer['z'] - layer['param']['arg']['stat1']) /
                                    tf.maximum(layer['param']['arg']['stat2'], tf.constant(1e-12, dtype=tf.float32)))
        elif normalization == Normalization.MAX:
            layer['h'] = layer['a']((layer['z'] - layer['param']['arg']['stat2']) /
                                    tf.maximum(layer['param']['arg']['stat1']-layer['param']['arg']['stat2'],
                                               tf.constant(1e-12, dtype=tf.float32)))
        else:
            layer['h'] = layer['a'](layer['z'])

        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_dense_layer(self, n, activation=tf.identity):
        layer = dict()
        layer['n'] = n
        layer['param'] = {'w': None, 'b': None, 'type': 'dense', 'arg': None}
        layer['param']['w'] = tf.Variable(tf.truncated_normal((self.layers[-1]['n'], layer['n']),
                                                              stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                          dtype=tf.float32,name='Layer'+str(len(self.layers))+'_W')
        layer['param']['b'] = tf.Variable(tf.zeros([layer['n']]),name='Layer'+str(len(self.layers))+'_B')

        bsize = tf.shape(self.layers[-1]['h'])[0]
        layer['z'] = tf.matmul(tf.reshape(self.layers[-1]['h'], [-1, self.layers[-1]['n']]), layer['param']['w']) + \
                     layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](tf.reshape(layer['z'],[bsize,-1,n]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
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

        bsize = tf.shape(self.layers[-1]['h'])[0]
        layer['z'] = tf.matmul(tf.reshape(self.layers[-1]['h'], [-1, self.layers[-1]['n']]),
                               tf.transpose(inv['param']['w'][:(-inv['n']),:])) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, inv['n']]))
        self.layers.insert(max(0, len(self.layers)), layer)
        return self

    def __init_gate(self, n, feeding_n, activation=tf.identity, name='gate'):
        with tf.variable_scope('rnn_cell'):
            gate = dict()
            gate['n'] = n
            gate['param'] = {'w': None, 'b': None, 'type': 'gate',
                              'arg': None}

            gate['param']['w'] = tf.get_variable(name + '_W', initializer=tf.truncated_normal(
                (feeding_n, gate['n']), stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                 dtype=tf.float32)
            gate['param']['b'] = tf.get_variable(name + '_B', (gate['n']), dtype=tf.float32,
                                                 initializer=tf.constant_initializer(0.0))

        gate['a'] = activation
        return gate

    def add_lstm_layer(self, n, use_last=False, peepholes=False, activation=tf.identity):
        self.recurrent = True
        self.use_last = use_last

        with tf.variable_scope('rnn_cell'):
            layer = dict()
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'recurrent',
                              'arg': {'timesteps': None, 'hsubt': None, 'cell': None}}

            layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                  initializer=tf.truncated_normal(
                                                      (self.layers[-1]['n'] + layer['n'], layer['n']),
                                                      stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))
            layer['param']['arg']['cell'] = tf.get_variable('Layer' + str(len(self.layers)) + '_C',
                                                            (layer['n']), dtype=tf.float32,
                                                            initializer=tf.constant_initializer(0.0))

        feeding_n = self.layers[-1]['n'] + n

        if peepholes:
            feeding_n += n

        forget_g = self.__init_gate(n, feeding_n, tf.sigmoid, name='Layer' + str(len(self.layers)) + '_forget')
        input_g = self.__init_gate(n, feeding_n, tf.sigmoid, name='Layer' + str(len(self.layers)) + '_input')
        output_g = self.__init_gate(n, feeding_n, tf.sigmoid, name='Layer' + str(len(self.layers)) + '_output')

        layer['a'] = activation
        L = self.layers

        def __lstm_step(state, input):
            with tf.variable_scope('rnn_cell', reuse=True):
                state = layer['a'](state)
                W = tf.get_variable('Layer' + str(len(L)) + '_W')
                W = tf.identity(W)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', W)

                b = tf.get_variable('Layer' + str(len(L)) + '_B')
                b = tf.identity(b)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', b)

                C = tf.get_variable('Layer' + str(len(L)) + '_C')
                C = tf.identity(C)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', C)

                # forget gate
                forgetW = tf.get_variable('Layer' + str(len(L)) + '_forget_W')
                forgetW = tf.identity(forgetW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', forgetW)

                forgetb = tf.get_variable('Layer' + str(len(L)) + '_forget_B')
                forgetb = tf.identity(forgetb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', forgetb)

                # input gate
                inputW = tf.get_variable('Layer' + str(len(L)) + '_input_W')
                inputW = tf.identity(inputW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', inputW)

                inputb = tf.get_variable('Layer' + str(len(L)) + '_input_B')
                inputb = tf.identity(inputb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', inputb)

                # output gate
                outputW = tf.get_variable('Layer' + str(len(L)) + '_output_W')
                outputW = tf.identity(outputW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', outputW)

                outputb = tf.get_variable('Layer' + str(len(L)) + '_output_B')
                outputb = tf.identity(outputb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', outputb)

                concat = tf.concat([tf.cast(tf.reshape(input, [-1, L[-1]['n']]), tf.float32), state], 1)
                cell_prime = tf.tanh(tf.matmul(concat, W) + b)

                p_concat = state if not peepholes else tf.concat([state, tf.cast(tf.reshape(
                    tf.tile(C, [tf.shape(state)[0]]), [-1, layer['n']]), tf.float32)], 1)

                concat_g = tf.concat([tf.cast(tf.reshape(input, [-1, L[-1]['n']]), tf.float32), p_concat], 1)

                forget_h = forget_g['a'](tf.matmul(concat_g, forgetW) + forgetb)
                input_h = input_g['a'](tf.matmul(concat_g, inputW) + inputb)

                C = (C * forget_h) + (cell_prime * input_h)

                pr_concat = state if not peepholes else tf.concat([state, C], 1)
                concat_g = tf.concat([tf.cast(tf.reshape(input, [-1, L[-1]['n']]), tf.float32), pr_concat], 1)
                output_h = output_g['a'](tf.matmul(concat_g, outputW) + outputb)

                layer['z'] = (output_h * tf.tanh(C))
                return layer['z']

        shape = tf.shape(self.layers[-1]['h'])
        init = tf.Variable(tf.zeros((1, layer['n'])))
        lstm_zs = tf.scan(__lstm_step, tf.transpose(self.layers[-1]['h'],[1,0,2]),
                         initializer=tf.reshape(tf.tile(init,[1, shape[0]]),[-1,layer['n']]))

        layer['h'] = layer['a'](tf.transpose(lstm_zs,[1,0,2]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.deepest_recurrent_layer = len(self.layers)-1
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_gru_layer(self, n, use_last=False, activation=tf.identity):
        self.recurrent = True
        self.use_last = use_last

        with tf.variable_scope('rnn_cell'):
            layer = dict()
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'recurrent',
                              'arg': {'timesteps': None, 'hsubt': None, 'cell': None}}

            layer['param']['w'] = tf.get_variable('Layer' + str(len(self.layers)) + '_W',
                                                  initializer=tf.truncated_normal(
                                                      (self.layers[-1]['n'] + layer['n'], layer['n']),
                                                      stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer' + str(len(self.layers)) + '_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))

        update_g = self.__init_gate(n, self.layers[-1]['n'] + n, activation=tf.sigmoid,
                                    name='Layer' + str(len(self.layers)) + '_update')
        reset_g = self.__init_gate(n, self.layers[-1]['n'] + n, activation=tf.sigmoid,
                                   name='Layer' + str(len(self.layers)) + '_reset')

        layer['a'] = activation
        L = self.layers

        def __gru_step(state, input):
            with tf.variable_scope('rnn_cell', reuse=True):
                state = layer['a'](state)
                W = tf.get_variable('Layer' + str(len(L)) + '_W')
                W = tf.identity(W)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', W)

                b = tf.get_variable('Layer' + str(len(L)) + '_B')
                b = tf.identity(b)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', b)

                updateW = tf.get_variable('Layer' + str(len(L)) + '_update_W')
                updateW = tf.identity(updateW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', updateW)

                updateb = tf.get_variable('Layer' + str(len(L)) + '_update_B')
                updateb = tf.identity(updateb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', updateb)

                resetW = tf.get_variable('Layer' + str(len(L)) + '_reset_W')
                resetW = tf.identity(resetW)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', resetW)

                resetb = tf.get_variable('Layer' + str(len(L)) + '_reset_B')
                resetb = tf.identity(resetb)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', resetb)

                concat_g = tf.concat([tf.cast(tf.reshape(input, [-1, L[-1]['n']]), tf.float32), state], 1)
                update_h = update_g['a'](tf.matmul(concat_g, updateW) + updateb)
                reset_h = reset_g['a'](tf.matmul(concat_g, resetW) + resetb)

                concat = tf.concat([tf.cast(tf.reshape(input, [-1,  L[-1]['n']]), tf.float32), reset_h * state], 1)
                cell_prime = tf.tanh(tf.matmul(concat, W) + b)

                layer['z'] = (1 - update_h) * state + update_h * cell_prime

                return layer['z']

        shape = tf.shape(self.layers[-1]['h'])
        init = tf.Variable(tf.zeros((1, layer['n'])))
        gru_zs = tf.scan(__gru_step, tf.transpose(self.layers[-1]['h'], [1, 0, 2]),
                          initializer=tf.reshape(tf.tile(init, [1, shape[0]]), [-1, layer['n']]))

        layer['h'] = layer['a'](tf.transpose(gru_zs, [1, 0, 2]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.deepest_recurrent_layer = len(self.layers)-1
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def add_rnn_layer(self, n, use_last=False, activation=tf.identity):
        self.recurrent = True
        self.use_last = use_last

        with tf.variable_scope('rnn_cell'):
            layer = dict()
            layer['n'] = n
            layer['param'] = {'w': None, 'b': None, 'type': 'recurrent',
                              'arg': {'init': None, 'hsubt': None, 'cell': None}}

            layer['param']['w'] = tf.get_variable('Layer'+str(len(self.layers))+'_W',
                                                  initializer=tf.truncated_normal(
                                                      (self.layers[-1]['n'] + layer['n'], layer['n']),
                                                      stddev=1. / np.sqrt(self.layers[-1]['n'])),
                                                  dtype=tf.float32)
            layer['param']['b'] = tf.get_variable('Layer'+str(len(self.layers))+'_B',
                                                  (layer['n']), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(0.0))
            layer['param']['arg']['init'] = tf.Variable(tf.zeros((1, layer['n'])),
                                                        name='Layer' + str(len(self.layers)) + '_init')

        layer['a'] = activation
        L = self.layers

        def __rnn_step(state, input):
            with tf.variable_scope('rnn_cell', reuse=True):
                state = layer['a'](state)
                W = tf.get_variable('Layer' + str(len(L)) + '_W')
                W = tf.identity(W)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'W', W)

                b = tf.get_variable('Layer' + str(len(L)) + '_B')
                b = tf.identity(b)
                tf.get_default_graph().add_to_collection('R' + str(len(L)) + 'b', b)

                concat = tf.concat(
                    [tf.cast(tf.reshape(input, [-1, L[-1]['n']]), tf.float32), state], 1)

                layer['z'] = tf.tanh(tf.matmul(concat, W) + b)
                return layer['z']

        shape = tf.shape(self.layers[-1]['h'])
        init = tf.Variable(tf.zeros((1, layer['n'])))
        rnn_zs = tf.scan(__rnn_step, tf.transpose(self.layers[-1]['h'], [1, 0, 2]),
                         initializer=tf.reshape(tf.tile(init, [1, shape[0]]), [-1, layer['n']]))

        layer['h'] = layer['a'](tf.transpose(rnn_zs, [1, 0, 2]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.deepest_hidden_layer = self.layers[-1]
        self.deepest_recurrent_layer = len(self.layers)-1
        self.__deepest_hidden_layer_ind = len(self.layers)-1
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

        bsize = tf.shape(self.layers[-1]['h'])[0]
        layer['z'] = tf.matmul(tf.nn.dropout(tf.reshape(self.layers[-1]['h'], [-1, self.layers[-1]['n']]), layer['param']['arg']),
                               layer['param']['w']) + layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, n]))
        self.layers.insert(max(0, len(self.layers)), layer)
        self.args[self.layers[-1]['param']['arg']] = keep
        self.deepest_hidden_layer = self.layers[-1]
        self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def __initialize(self, cost_function=Cost.MSE):
        if not self.__is_init:
            self.y = tf.placeholder(tf.float32, [None, None, self.layers[-1]['n']], name='y')

            if cost_function == Cost.CROSS_ENTROPY:
                self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.layers[-1]['h']),
                                                                   reduction_indices=[-1]))
            elif cost_function == Cost.L2_NORM:
                self.cost_function = tf.reduce_sum((self.layers[-1]['h'] - self.y)**2, reduction_indices=[-1])/2
            elif cost_function == Cost.RMSE:
                self.cost_function = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.layers[-1]['h'], self.y),
                                                            reduction_indices=[-1]))
            else:
                self.cost_function = tf.reduce_mean(tf.squared_difference(self.layers[-1]['h'], self.y),
                                                    reduction_indices=[-1])

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.layers[-1]['h'], 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.update = tf.train.AdagradOptimizer(self.step_size, name='update')
            self.minimize_cost = self.update.minimize(self.cost_function)

            self.var_grads = self.update.compute_gradients(self.cost_function, tf.trainable_variables())
            self.clipped_var_grads = [(tf.clip_by_norm(grad if grad is not None else tf.zeros_like(var), 10.), var) for
                                      grad, var in self.var_grads]

            self.update_weights = self.update.apply_gradients(self.clipped_var_grads)

            tf.global_variables_initializer().run()

            tf.get_default_graph().finalize()

            self.__is_init = True

    def __backprop_through_time(self, x, y, s):
        batch_cost = []

        valid = np.argwhere(np.array([len(k) for k in x[s]]) > 0).ravel()

        series_batch = x[s][valid]
        series_label = None
        if not self.use_last:
            series_label = y[s][valid]
        n_timestep = max([len(k) for k in series_batch])

        series_batch_padded = np.array(
            [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])
        series_label_padded = np.array(
            [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_label])

        self.args[self.layers[0]['z']] = series_batch_padded.reshape((len(series_batch),n_timestep,-1))
        self.args[self.y] = series_label_padded.reshape((len(series_batch), n_timestep, -1))

        cost, _ = self.session.run([self.cost_function, self.update_weights], feed_dict=self.args)
        batch_cost.append(cost)

        return batch_cost

    def train(self, x, y, validation_data=None, validation_labels=None, step=0.1, max_epochs=100, threshold=0.01, batch=10, cost_method=Cost.MSE):

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

        if self.normalization == Normalization.Z_SCORE:
            self.args[self.layers[0]['param']['arg']['stat1']] = np.mean(np.hstack([np.array(i).ravel() for i in x])
                                                                         .reshape((-1, np.array(x[0]).shape[1])),
                                                                         axis=0).reshape((-1))
            self.args[self.layers[0]['param']['arg']['stat2']] = np.std(np.hstack([np.array(i).ravel() for i in x])
                                                                        .reshape((-1, np.array(x[0]).shape[1])),
                                                                        axis=0).reshape((-1))
        elif self.normalization == Normalization.MAX:
            self.args[self.layers[0]['param']['arg']['stat1']] = np.max(np.hstack([np.array(i).ravel() for i in x])
                                                                        .reshape((-1, np.array(x[0]).shape[1])),
                                                                        axis=0).reshape((-1))
            self.args[self.layers[0]['param']['arg']['stat2']] = np.min(np.hstack([np.array(i).ravel() for i in x])
                                                                        .reshape((-1, np.array(x[0]).shape[1])),
                                                                        axis=0).reshape((-1))

        train_start = time.time()
        current_best_auc = 0;
        use_validation = False

        if validation_labels is not None and validation_data is not None:
            use_validation = True
            v_labels = np.array(flatten_sequence(validation_labels, True))

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
                    self.minimize_cost.run(feed_dict=self.args)
                    cost.append(self.getCost(x[s], y[s], False))

            if e > 1:
                mean_last_ten = np.mean(cost[-10:])
            else:
                mean_last_ten = 0

            print("{:<10}{:^10.4f}{:>9.1f}s".format("Epoch " + str(e), np.mean(cost),
                                                    time.time() - epoch_start))

            if use_validation:
                v_predictions = np.array(flatten_sequence(self.predict(validation_data), True))
                current_auc = auc(v_labels,v_predictions)
                if current_auc >= 0.5 : # current_best_auc:
                    current_best_auc = current_auc
                else:
                    break;
            else:
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
            pred = []

            valid = np.argwhere(np.array([len(k) for k in x]) > 0).ravel()
            series_batch = x[valid]

            samp_timestep = [len(k) for k in series_batch]
            n_timestep = max(samp_timestep)

            series_batch_padded = np.array(
                [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])

            arg[self.layers[0]['z']] = series_batch_padded.reshape((len(x), n_timestep, -1))
            p = self.session.run(self.layers[-1]['h'],feed_dict=arg)

            for j in range(len(samp_timestep)):
                pred.append(np.array(p[j])[:samp_timestep[j]])

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


def run_sine_test():
    np.random.seed(1)
    data, labels = generate_timeseries_test(samples=1000, max_sequence=20, regular_offset=0.3, categorical=False)

    validation_data, validation_labels = generate_timeseries_test(samples=100, max_sequence=20, regular_offset=0.3, categorical=False)

    np.random.seed(0)
    tf.set_random_seed(0)

    net = Network() \
        .add_input_layer(data[0].shape[-1]) \
        .add_rnn_layer(2, activation=tf.identity) \
        .add_dense_layer(labels[0].shape[-1], activation=tf.identity)

    #net.set_max_backprop_timesteps(3)

    net.train(data, labels,validation_data, validation_labels, max_epochs=50, step=1e-3, batch=10, cost_method='rmse', threshold=0.01)

    pred = net.predict(np.sin(np.array(range(20)) * 0.3).reshape((1, -1, 1)))

    fp = np.array(flatten_sequence(pred, True))
    fp[:, 0] = np.sin((np.array(range(20))+1) * 0.3)
    writetoCSV(fp, 'predictions')


def run_npstopout_test():
    np.random.seed(1)
    data, labels = generate_timeseries_test(samples=1000, max_sequence=20, regular_offset=0.3, categorical=False)

    print(labels.shape)
    print(labels[0].shape)

    training = np.load('resources/nps_training.npy', encoding='bytes')
    training_labels = np.load('resources/nps_training_labels.npy', encoding='bytes')

    validation_data = np.load('resources/nps_testing.npy', encoding='bytes')
    validation_labels = np.load('resources/nps_testing_labels.npy', encoding='bytes')

    testing = np.load('resources/nps_testing.npy', encoding='bytes')
    testing_labels = np.load('resources/nps_testing_labels.npy', encoding='bytes')

    net = Network() \
        .add_input_layer(training[0].shape[-1], normalization=Normalization.Z_SCORE) \
        .add_lstm_layer(200, peepholes=True, activation=tf.nn.relu) \
        .add_dense_layer(training_labels[0].shape[-1], activation=tf.nn.softmax)

    net.train(x=training[:200], y=training_labels[:200],
              step=0.01,
              max_epochs=20,
              threshold=1e-3,
              batch=3,
              cost_method=Cost.CROSS_ENTROPY,
              validation_data=validation_data,
              validation_labels=validation_labels)

    pred = net.predict(testing[:200])

    fp = np.array(flatten_sequence(pred, True))
    fl = np.array(flatten_sequence(testing_labels[:200], True))

    for i in range(training_labels[0].shape[-1]):
        fp = np.insert(fp, fp.shape[-1], fl[:, i], axis=1)

    writetoCSV(fp, 'nps_predictions')

    print('AUC: {}'.format(auc(fl[:,1:4],fp[:,1:4])))


def run_scan_test():
    sess = tf.InteractiveSession()

    x = [[[1],[2],[3],[4]],[[10],[2],[3],[4]],[[100],[2],[3],[4]]]
    print(x)

    z = tf.placeholder(tf.float32, [None,4, 1])

    def sum(a, x):
        return a+x

    tf_sum = tf.scan(sum, z)

    d = dict()
    d[z] = np.array(x).reshape(-1,4,1)

    print('===========================')
    print(sess.run(tf_sum, feed_dict=d))


if __name__ == "__main__":
    # run_sine_test()

    run_npstopout_test()
    # run_scan_test()

    # data = du.read_csv('nps_predictions.csv',headers=False)
    # for i in data[:10]:
    #     print(i)
