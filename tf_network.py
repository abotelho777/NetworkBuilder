import numpy as np
import tensorflow as tf
from datautility import deprecated
import datautility as du
import evaluationutility as eu
import time


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

        self.cost_method = Cost.MSE
        self.cost_function = None
        self.__cost = []
        self.__outputs = []
        self.__output_layers = []
        self.__output_weights = []

        self.__tmp_multi_out = None

        self.deepest_hidden_layer = None

        self.recurrent = False
        self.use_last = False
        self.deepest_recurrent_layer = None

        self.__deepest_hidden_layer_ind = None

        self.graph = tf.get_default_graph()

        # self.session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))  # use CPU
        self.session = tf.InteractiveSession()  # use GPU

    def set_default_cost_method(self, cost_method=Cost.MSE):
        self.cost_method = cost_method

    def get_deepest_hidden_layer(self):
        return self.deepest_hidden_layer

    def get_deepest_hidden_layer_index(self):
        return self.__deepest_hidden_layer_ind

    def get_layer(self, index):
        if not (0 <= index < len(self.layers)):
            raise IndexError('Index is out of range for the network with {} layers'.format(len(self.layers)))
        return self.layers[index]

    def begin_multi_output(self, cost_methods=None, weights=None):
        self.__tmp_multi_out = dict()

        if cost_methods is None:
            cost_methods = [self.cost_method]
        cost_methods = np.array(cost_methods).reshape((-1))

        if weights is None:
            weights = [1]
        weights = np.array(weights).reshape((-1))

        self.__tmp_multi_out['methods'] = cost_methods
        self.__tmp_multi_out['weights'] = weights
        self.__tmp_multi_out['deepest_hidden'] = self.__deepest_hidden_layer_ind

        return self

    def end_multi_output(self):
        if self.__tmp_multi_out is None:
            return self

        self.__deepest_hidden_layer_ind = self.__tmp_multi_out['deepest_hidden']
        self.deepest_hidden_layer = self.layers[self.__tmp_multi_out['deepest_hidden']]

        for i in range(self.__tmp_multi_out['deepest_hidden']+1, len(self.layers)):
            if len(self.__tmp_multi_out['methods']) == 1:
                m = self.__tmp_multi_out['methods'][0]
            elif i < len(self.__tmp_multi_out['methods']):
                m = self.__tmp_multi_out['methods'][i]
            else:
                m = self.cost_method

            if len(self.__tmp_multi_out['weights']) == 1:
                w = self.__tmp_multi_out['weights'][0]
            elif i < len(self.__tmp_multi_out['weights']):
                w = self.__tmp_multi_out['weights'][i]
            else:
                w = 1

            self.__outputs.append(self.layers[i]['h'])
            self.__output_layers.append(i)
            self.__cost.append(m)
            self.__output_weights.append(w)

        self.__tmp_multi_out = None
        return self

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

    def add_input_layer_from_network(self, network, layer_index):
        network_layer = network.get_layer(layer_index)

        layer = dict()
        layer['n'] = network_layer['n']
        layer['z'] = network.layers[0]['z']
        layer['param'] = network_layer['param']
        layer['a'] = network_layer['a']
        layer['h'] = network_layer['h']

        self.normalization = Normalization.NONE
        self.args = network.args
        self.recurrent = network.recurrent

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
        if self.__tmp_multi_out is None:
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
                               tf.transpose(inv['param']['w'])) + \
                     layer['param']['b']
        layer['a'] = activation
        layer['h'] = layer['a'](tf.reshape(layer['z'], [bsize, -1, self.layers[layer_index-1]['n']]))
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
        self.deepest_recurrent_layer = len(self.layers)-1

        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
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
        self.deepest_recurrent_layer = len(self.layers)-1

        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
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
        self.deepest_recurrent_layer = len(self.layers)-1

        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
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
        if self.__tmp_multi_out is None:
            self.deepest_hidden_layer = self.layers[-1]
            self.__deepest_hidden_layer_ind = len(self.layers) - 1
        return self

    def __initialize(self):
        if not self.__is_init:

            self.y = []

            if len(self.__outputs) == 0:
                self.__outputs.append(self.layers[-1]['h'])
                self.__output_layers.append(len(self.layers)-1)
                self.__cost.append(self.cost_method)
                self.__output_weights.append(1)
                self.__deepest_hidden_layer_ind -= 1
                self.deepest_hidden_layer = self.layers[self.__deepest_hidden_layer_ind]

            for i in range(len(self.__outputs)):

                out_y = tf.placeholder(tf.float32, [None, None, self.layers[self.__output_layers[i]]['n']],
                                       name='y'+str(i))

                method = self.__cost[i]

                if method == Cost.CROSS_ENTROPY:
                    sum_cross_entropy = -tf.reduce_sum(
                        tf.where(tf.is_nan(out_y), self.__outputs[i], out_y) * tf.log(self.__outputs[i]),
                        reduction_indices=[-1])
                    sce = tf.reduce_sum(tf.where(tf.is_nan(sum_cross_entropy), tf.zeros_like(sum_cross_entropy),
                                                 sum_cross_entropy))
                    cost_fn = sce/(tf.cast(tf.count_nonzero(sum_cross_entropy),
                                           tf.float32)+tf.constant(1e-8, dtype=tf.float32))

                elif method == Cost.L2_NORM:
                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = sse / 2

                elif method == Cost.RMSE:
                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = tf.sqrt(sse/(tf.cast(tf.count_nonzero(sq_dif),
                                                   tf.float32)+tf.constant(1e-8, dtype=tf.float32)))


                else:
                    sq_dif = tf.squared_difference(self.__outputs[i], tf.where(tf.is_nan(out_y),
                                                                               self.__outputs[i], out_y))
                    sse = tf.reduce_sum(tf.reduce_sum(tf.where(tf.is_nan(sq_dif), tf.zeros_like(sq_dif), sq_dif),
                                                      reduction_indices=[-1]))
                    cost_fn = sse / (tf.cast(tf.count_nonzero(sq_dif),
                                             tf.float32) + tf.constant(1e-8, dtype=tf.float32))

                if self.cost_function is None:
                    self.cost_function = self.__output_weights[i] * cost_fn
                else:
                    self.cost_function += self.__output_weights[i] * cost_fn

                self.y.append(out_y)

            self.update = tf.train.AdagradOptimizer(self.step_size, name='update')
            self.minimize_cost = self.update.minimize(self.cost_function)

            self.var_grads = self.update.compute_gradients(self.cost_function, tf.trainable_variables())
            self.clipped_var_grads = [(tf.clip_by_norm(
                tf.where(tf.is_nan(grad if grad is not None else tf.zeros_like(var)), tf.zeros_like(var),
                         grad if grad is not None else tf.zeros_like(var)), 10.), var) for grad, var in self.var_grads]
            self.update_weights = self.update.apply_gradients(self.clipped_var_grads)

            tf.global_variables_initializer().run()

            # tf.get_default_graph().finalize()

            self.__is_init = True

    def __backprop_through_time(self, x, y, s):
        batch_cost = []

        valid = np.argwhere(np.array([len(k) for k in x[s]]) > 0).ravel()

        series_batch = x[s][valid]

        # print(len(y))
        # print(np.array(y, dtype=object))
        # exit(1)
        series_label = None
        if not self.use_last:
            series_label = y[s][valid]
        n_timestep = max([len(k) for k in series_batch])

        series_batch_padded = np.array(
            [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])

        series_label_padded = []

        # print(series_label)
        # print(np.array([a[i] for a in series_label]))
        # exit(1)

        for i in range(len(series_label[0])):
            series_label_padded.append(np.array(
                [np.pad(sl, ((0, n_timestep - len(sl)), (0, 0)), 'constant', constant_values=np.nan) for sl in
                 np.array([a[i] for a in series_label])]).reshape((len(series_batch), n_timestep, -1)))

        # print(series_label_padded)
        # exit(1)

        self.args[self.layers[0]['z']] = series_batch_padded.reshape((len(series_batch), n_timestep, -1))

        if not len(self.y) == len(series_label_padded):
            raise IndexError('The number of output layers does not match the labels supplied')

        for i in range(len(self.y)):
            self.args[self.y[i]] = series_label_padded[i]

        # print(self.session.run(self.layers[0]['h'], feed_dict=self.args))
        # exit(1)

        cost, _ = self.session.run([self.cost_function, self.update_weights], feed_dict=self.args)
        # print(cost)
        # if cost == float('nan') or np.isnan(cost):
        # print('output',self.session.run(self.__outputs,feed_dict=self.args))
        # print('labels',series_label_padded)
        # print('maybe',self.session.run(self.maybe,feed_dict=self.args))
        # print('cost',self.session.run(self.cost_function, feed_dict=self.args))
        # print('cost n', self.session.run(self.cost_n,feed_dict=self.args))
        # exit(1)
        batch_cost.append(cost)

        return batch_cost

    def train(self, x, y, validation_data=None, validation_labels=None, step=0.1, max_epochs=100, threshold=0.01,
              batch=1, cost_method=Cost.MSE):

        if not (du.ndims(x) == 3 and du.ndims(y) == 4):
            pass
            # TODO: if data is passed as wrong shape, reformat
            # TODO: ensure validation data also has correct format/shape

        use_validation = False

        if validation_labels is not None and validation_data is not None:
            use_validation = True
            # v_labels = np.array(flatten_sequence(validation_labels, True))

        self.step_size = step
        self.batch_size = batch
        self.training_epochs = max_epochs

        self.__initialize()

        print("{:=<40}".format(''))
        print("{:^40}".format("Training Network"))
        print("{:=<40}".format(''))
        structure = "{}n".format(self.layers[0]['n'])
        for i in range(1, self.__deepest_hidden_layer_ind + 1):
            structure += " -> {}n".format(self.layers[i]['n'])
        structure += " -> {}n".format(self.layers[self.__output_layers[0]]['n'])
        for i in range(1, len(self.__outputs)):
            structure += ", {}n".format(self.layers[self.__output_layers[i]]['n'])

        print("-{} layers: {}".format(len(self.layers), structure))
        print("-{} epochs".format(max_epochs))
        print("-step size = {}".format(step))
        print("-batch size = {}".format(batch))
        print("{:=<40}".format(''))
        # TODO: include validation cost when using validation set
        print("{:<10}{:^10}{:>10}".format("Epoch", "Cost", "Time"))
        print("{:=<40}".format(''))

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

        e = 1
        while True:
            epoch_start = time.time()

            # v = list(range(x.shape[0]))
            # np.random.shuffle(v)
            # x = x[v]
            # y = y[v]

            cost = []
            for i in range(0, x.shape[0], batch):
                s = np.array(range(i, min(x.shape[0], i + batch)))
                if len(s) < batch:
                    continue

                if self.recurrent:
                    batch_cost = self.__backprop_through_time(x, y, s)
                    for j in batch_cost:
                        cost.append(j)
                else:
                    ys = []

                    for j in range(len(y[s][0])):
                        ys.append(np.array([a[j] for a in y[s]]).reshape((batch, 1, -1)))

                    self.args[self.layers[0]['z']] = x[s].reshape((batch, 1, -1))

                    if not len(self.y) == len(ys):
                        raise IndexError('The number of output layers does not match the labels supplied')

                    for j in range(len(self.y)):
                        self.args[self.y[j]] = ys[j]

                    self.minimize_cost.run(feed_dict=self.args)
                    cost.append(self.get_cost(x[s], y[s], False))

            if e > 1:
                mean_last_ten = np.mean(cost[-10:])
            else:
                mean_last_ten = 0


            # TODO: print validation cost in addition to training when using validation set
            print("{:<10}{:^10.4f}{:>9.1f}s".format("Epoch " + str(e), np.nanmean(cost),
                                                    time.time() - epoch_start))

            # TODO: use validation set in stopping criterion when using validation set
            # if use_validation:
            #     v_predictions = np.array(flatten_sequence(self.predict(validation_data), True))
            #     current_auc = auc(v_labels,v_predictions)
            #     if current_auc >= 0.5 : # current_best_auc:
            #         current_best_auc = current_auc
            #     else:
            #         break
            # else:
            #     if (0.0001 < abs(np.mean(cost[-10:]) - mean_last_ten) < threshold) or e >= max_epochs:
            #         break

            if (0.0001 < abs(np.mean(cost[-10:]) - mean_last_ten) < threshold) or e >= max_epochs:
                break
            e += 1

        print("{:=<40}".format(''))
        print("Total Time: {:<.1f}s".format(time.time() - train_start))

    def predict(self, x):
        # TODO: add ability to predict from last hidden layer

        if not (du.ndims(x) == 3):
            pass
            # TODO: if data is passed as wrong shape, reformat

        arg = dict(self.args)
        for i in arg:
            if 'keep' in i.name:
                arg[i] = 1
        for i in self.y:
            del arg[i]

        if self.recurrent:
            pred = []

            valid = np.argwhere(np.array([len(k) for k in x]) > 0).ravel()
            series_batch = x[valid]

            samp_timestep = [len(k) for k in series_batch]
            n_timestep = max(samp_timestep)

            series_batch_padded = np.array(
                [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])

            arg[self.layers[0]['z']] = series_batch_padded.reshape((len(x), n_timestep, -1))
            p = self.session.run(self.__outputs, feed_dict=arg)

            for i in range(len(self.__outputs)):
                out_p = []
                for j in range(len(samp_timestep)):
                    out_p.append(np.array(p[i][j])[:samp_timestep[j]])
                pred.append(np.array(out_p))

            return pred
        else:
            arg[self.layers[0]['z']] = x
            out = self.session.run(self.__outputs, feed_dict=arg)
            return [flatten_sequence(a) for a in out]

    def get_cost(self, x, y, test=True):

        if not (du.ndims(x) == 3 and du.ndims(y) == 4):
            pass
            # TODO: if data is passed as wrong shape, reformat

        arg = dict(self.args)
        if test:
            for i in arg:
                if 'keep' in i.name:
                    arg[i] = 1


        if self.recurrent:
            pred = []

            valid = np.argwhere(np.array([len(k) for k in x]) > 0).ravel()
            series_batch = x[valid]

            samp_timestep = [len(k) for k in series_batch]
            n_timestep = max(samp_timestep)

            series_batch_padded = np.array(
                [np.pad(sb, ((0, n_timestep - len(sb)), (0, 0)), 'constant', constant_values=0) for sb in series_batch])

            arg[self.layers[0]['z']] = series_batch_padded.reshape((len(x), n_timestep, -1))
        else:
            arg[self.layers[0]['z']] = x

        return self.cost_function.eval(feed_dict=arg)


@deprecated
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


@deprecated
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


@deprecated
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


@deprecated
def loadCSVwithHeaders(filename, max_rows=None):
    if max_rows is not None:
        max_rows += 1
    data = loadCSV(filename,max_rows)
    headers = np.array(data[0])
    # data = np.array(convert_to_floats(data))
    data = np.delete(data, 0, 0)
    return data,headers


@deprecated
def readHeadersCSV(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            # split out each comma-separated value
            return line.strip().split(',')
    return []


@deprecated
def softmax(z):
    t = np.exp(z - z.max())
    return t / np.sum(t, axis=1, keepdims=True)


def print_label_distribution(labels, label_names=None):
    # TODO: rewrite with new structure
    # TODO: move to datautility

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


def flatten_sequence(sequence):
    # TODO: move to datautility

    seq = list(sequence)
    dims = du.ndims(seq)

    if dims <= 2:
        return seq

    try:  # try the simple case
        return np.hstack([np.array(i).ravel() for i in seq]).reshape((-1, np.array(seq[0]).shape[1]))
    except (ValueError, IndexError):
        try:
            ar = None
            for i in range(len(seq)):
                row = seq[i][0]
                for t in range(1, len(seq[i])):
                    ntime = len(seq[i][t])
                    row = np.append(row, np.hstack([np.array(j).ravel() for j in seq[i][t]]).reshape((ntime, -1)), 1)

                if ar is None:
                    ar = row
                else:
                    ar = np.append(ar, row, 0)
            return ar
        except ValueError:
            raise ValueError('sequence must be in the basic shape: (sample, time step, ... )')


def format_data(table, identifier=None, labels=None, columns=None, order=None, as_sequence=False):
    # TODO: move to datautility
    if as_sequence:
        if identifier is None:
            raise ValueError('identifier cannot be None when formatting as a sequence.')
        return reshape_sequence(table, identifier, labels, columns, order)

    table = np.array(table)
    table[np.where(table == '')] = 'nan'

    if order is None:
        ordering = list(range(len(table)))
    else:
        try:
            tbl_order = np.array(table[:, order], dtype=np.float32)
        except ValueError:
            tbl_order = np.array(table[:, order], dtype=str)
        ordering = np.argsort(tbl_order)

    table = table[ordering]

    id_ind = table.shape[1]
    table = np.append(table, np.array(range(len(table))).reshape((-1, 1)), 1)

    return reshape_sequence(table, id_ind, labels, columns, None)


def reshape_sequence(table, pivot, labels=None, columns=None, order=None):
    # TODO: move to datautility
    if columns is None:
        columns = range(table.shape[-1])
    col = np.array(columns)

    table = np.array(table)
    table[np.where(table == '')] = 'nan'

    try:
        tbl_order = np.array(table[:, pivot], dtype=np.float32)
    except ValueError:
        tbl_order = np.array(table[:, pivot], dtype=str)

    _, piv = np.unique(tbl_order, return_index=True)
    p = table[piv, pivot]

    seq = dict()
    key = []
    x = []
    y = []

    for i in p:
        match = np.argwhere(np.array(table[:, pivot]) == i).ravel()

        key.append(i)

        if order is None:
            ordering = list(range(len(match)))
        else:
            try:
                tbl_order = np.array(table[match, order], dtype=np.float32)
            except ValueError:
                tbl_order = np.array(table[match, order], dtype=str)
            ordering = np.argsort(tbl_order)

        x.append(np.array(table[match, col.reshape((-1, 1))],
                                 dtype=np.float32).T[ordering].reshape((-1, len(col))))

        lab = None
        if labels is not None:
            labels = np.array(labels)

            if not hasattr(labels[0], '__iter__'):
                lab = {0: np.array(table[match, np.array(labels).reshape((-1, 1))],
                               dtype=np.float32).T[ordering].reshape((-1, len(labels)))}
            else:
                lab = dict()
                ind = 0
                for j in labels:
                    mlabel = np.array(j).reshape((-1))
                    lab[ind] = np.array(table[match, np.array(mlabel).reshape((-1, 1))],
                                          dtype=np.float32).T[ordering].reshape((-1, len(mlabel)))
                    ind += 1

        y.append(lab)
    seq['key'] = np.array(key)
    seq['x'] = np.array(x)
    seq['y'] = np.array(y)
    return seq

@deprecated
def Aprime(actual, predicted):
    assert len(actual) == len(predicted)

    # print(actual[0:15])
    #
    # print(predicted[0:15])

    score = [[],[]]

    for i in range(0,len(actual)):
        # print(actual[i])
        # print(predicted[i])
        if not np.isnan(actual[i]):
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
    data, labels = generate_timeseries_test(samples=1000, max_sequence=10, regular_offset=None, categorical=False)

    np.random.seed(0)
    tf.set_random_seed(0)

    net = Network() \
        .add_input_layer(data[0].shape[-1]) \
        .add_rnn_layer(2, activation=tf.identity) \
        .add_dense_layer(labels[0].shape[-1], activation=tf.identity)

    net.set_default_cost_method(Cost.MSE)

    net.train(data, labels, step=0.01,
              max_epochs=20,
              threshold=1e-3,
              batch=1)

    pred = net.predict(np.sin(np.array(range(20)) * 0.3).reshape((1, -1, 1)))

    fp = np.array(flatten_sequence(pred, True))
    fp[:, 0] = np.sin((np.array(range(20))+1) * 0.3)
    writetoCSV(fp, 'predictions')


def run_npstopout_test():
    np.random.seed(1)
    data, labels = generate_timeseries_test(samples=1000, max_sequence=5, regular_offset=0.3, categorical=False)

    print(labels.shape)
    print(labels[0].shape)

    training = np.load('resources/nps_training.npy', encoding='bytes')
    training_labels = np.load('resources/nps_training_labels.npy', encoding='bytes')

    testing = np.load('resources/nps_testing.npy', encoding='bytes')
    testing_labels = np.load('resources/nps_testing_labels.npy', encoding='bytes')

    net = Network() \
        .add_input_layer(training[0].shape[-1], normalization=Normalization.Z_SCORE) \
        .add_lstm_layer(200, peepholes=True, activation=tf.nn.relu) \
        .begin_multi_output(cost_methods=[Cost.CROSS_ENTROPY]) \
        .add_dense_layer(training_labels[0].shape[-1], activation=tf.nn.softmax) \
        .end_multi_output()

    net.set_default_cost_method(Cost.CROSS_ENTROPY)

    net.train(x=training[:200], y=training_labels[:200],
              step=0.01,
              max_epochs=20,
              threshold=1e-3,
              batch=3)

    pred = net.predict(testing[:200])

    fp = np.array(flatten_sequence(pred, True))
    fl = np.array(flatten_sequence(testing_labels[:200], True))

    for i in range(training_labels[0].shape[-1]):
        fp = np.insert(fp, fp.shape[-1], fl[:, i], axis=1)

    writetoCSV(fp, 'nps_predictions')

    print('AUC: {}'.format(eu.auc(fl[:,1:4],fp[:,1:4])))


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


def run_multi_label_test():
    # data, labels = du.read_csv('resources/artificial_sequences.csv')
    # du.print_descriptives(data,labels)
    #
    # flat = format_data(data,1, [2,3,4], [2,3,4], 0, False)
    # seq = format_data(data, 1, [[5], [6, 7, 8]], [2, 3, 4], 0, True)
    # # seq = reshape_sequence(data, 1, [[5],[6,7,8]], [2,3,4],0)

    ################
    #maxrows = 100000

    #data, labels = du.read_csv('labeled_compressed92features.csv',max_rows=maxrows)
    # data, labels = du.read_csv('labeled_compressed92features.csv')
    # du.print_descriptives(data, labels)
    #
    # flat = format_data(data, 1, list(range(4,96)),list(range(4,96)),3, False)
    # seq = format_data(data, 1, [100],list(range(4,96)),3, True)
    # #Wheelspin 100
    #
    # np.save('x_formatted_wheelspin.npy', seq['x'])
    # np.save('y_formatted.npy_wheelspin', seq['y'])
    #
    # np.save('x_flat_wheelspin.npy', flat['x'])
    # np.save('y_flat_wheelspin.npy', flat['y'])
    # ###################
    rows = 100
    seq = dict()
    # x = []
    # y = []
    x = np.load('x_formatted.npy')
    y = np.load('y_formatted.npy')
    # x = np.load('x_formatted_wheelspin.npy')
    # y = np.load('y_formatted.npy_wheelspin.npy')
    seq['x'] = np.array(x)[:rows]
    seq['y'] = np.array(y)[:rows]
    seq['y'] = myOffset(seq['y'])
    ############################

    net = Network().add_input_layer(92,normalization=Normalization.NONE)\
        .add_rnn_layer(200,activation=tf.nn.relu)\
        .add_dense_layer(1, activation=tf.nn.sigmoid)

    net.set_default_cost_method(Cost.MSE)

    net.train(x=seq['x'], y=seq['y'], step=0.01,
              max_epochs=20, threshold=0.0001, batch=2)

    pred = net.predict(x=seq['x'])

    ###############################
    # rows = 10
    # seq = dict()
    # flat = dict()
    # # x = []
    # # y = []
    # x = np.load('x_formatted.npy')
    # y = np.load('y_formatted.npy')
    # fx = np.load('x_flat.npy')
    # fy = np.load('y_flat.npy')
    # seq['x'] = np.array(x)[:rows]
    # seq['y'] = np.array(y)[:rows]
    # flat['x'] = np.array(fx)[:rows]
    # flat['y'] = np.array(fy)[:rows]
    # # print('seq["y"][0]',seq['y'][0])
    # # print('seq["y"][0]',seq['y'][1])
    # seq['y'] = myOffset(seq['y'])
    # #flat['y'] = myOffset(flat['y'])
    # # print('Offset', seq['y'][0])
    # # print('Offset', seq['y'][1])
    #
    # ae = Network().add_input_layer(92, normalization=Normalization.NONE)\
    #     .add_dense_layer(46, activation=tf.nn.tanh)\
    #     .add_inverse_layer(layer_index=-1, activation=tf.nn.sigmoid)
    # ae.set_default_cost_method(Cost.L2_NORM)
    # print('DBaeTrain')
    # ae.train(x=flat['x'], y=flat['y'], step=0.01,max_epochs=2,threshold=0.0001,batch=2)
    #
    # net = Network().add_input_layer_from_network(ae, ae.get_deepest_hidden_layer_index())\
    #     .add_rnn_layer(46, activation=tf.nn.relu)\
    #     .begin_multi_output([Cost.RMSE, Cost.CROSS_ENTROPY])\
    #     .add_dense_layer(1, activation=tf.nn.sigmoid) \
    #     .end_multi_output()
    #
    # net.set_default_cost_method(Cost.CROSS_ENTROPY)
    # print('DBTrain')
    # net.train(x=seq['x'], y=seq['y'], step=0.01,
    #           max_epochs=2, threshold=0.0001, batch=2)
    # print('DBPred')
    # pred = net.predict(x=seq['x'])

    # net = Network().add_input_layer_from_network(ae, ae.get_deepest_hidden_layer_index())\
    #     .add_rnn_layer(10, activation=tf.nn.relu)\
    #     .begin_multi_output([Cost.RMSE, Cost.CROSS_ENTROPY])\
    #     .add_dense_layer(1, activation=tf.nn.sigmoid) \
    #     .add_dense_layer(3, activation=tf.nn.softmax)\
    #     .end_multi_output()

    # print('========== PREDICTIONS ==========')
    # p = 0
    # for i in pred:
    #     print('\n------ Y{} ------'.format(p))
    #     flat_p = flatten_sequence(i)
    #     for j in flat_p:
    #         print(j)
    #     p += 1

    # print('--',flatten_sequence(pred[0]).ravel())
    #print(eu.auc(actual=np.array(data[:, 97],dtype=np.float32),predicted=flatten_sequence(pred[0]).ravel()))
    #print(flatten_sequence(pred[0]).ravel())
    # a = my4dto2d(seq['y'])
    # print(a[0])
    # print(eu.auc(actual=a,predicted=flatten_sequence(pred[0]).ravel()))
    # print('DBAUC')
    # print(eu.auc(actual=my4dto2d(seq['y']),predicted=flatten_sequence(pred[0]).ravel()))
    #print(eu.auc(actual=np.array(data[:, 97],dtype=np.float32),predicted=flatten_sequence(pred[0]).ravel()))
    print(Aprime(actual=my4dto2d(seq['y']),predicted=flatten_sequence(pred[0]).ravel()))




def my4dto2d(array):
    result = []
    for i in array:
        result.extend((i[0]).ravel())
    return np.array(result,dtype=np.float32)
    #return result



def myOffset(y,label=0):
    result = np.array(y)

    for s in range(len(result)):
        for t in range(len(result[s][label])-1):
            result[s][label][t] = result[s][label][t+1]
        result[s][label][-1] = np.ones_like(result[s][label][0])*np.nan
    return result


def run_npc_test(lb):
    outputlabel = lb
    haveAE = False

    seq = dict()
    n_folds = 5

    x = np.load('seq_x_' + outputlabel + '.npy')
    y = np.load('seq_y_' + outputlabel + '.npy')
    seq['x'] = np.array(x)
    seq['y'] = np.array(y)

    if(haveAE):
        k = np.load('seq_k_' + outputlabel + '.npy')
        seq['key'] = np.array(k)

        flatk = np.load('flat_k_' + outputlabel + '.npy')
        flatx = np.load('flat_x_' + outputlabel + '.npy')
        flaty = np.load('flat_y_' + outputlabel + '.npy')

        flat = dict()
        flat['key'] = np.array(flatk)
        flat['x'] = np.array(flatx)
        flat['y'] = np.array(flaty)

    if outputlabel == 'npc' or outputlabel == 'fa':
        seq['y'] = myOffset(seq['y'])

    tf.set_random_seed(0)
    np.random.seed(0)

    fold = np.random.randint(0, n_folds, len(seq['x']))

    fold_auc = []
    fold_rmse = []

    for i in range(n_folds):
        tf.reset_default_graph()

        training = np.argwhere(fold != i).ravel()
        test_set = np.argwhere(fold == i).ravel()

        if(haveAE):
            T = seq['key'][training]
            aetraining = []
            for tt in T:
                aetraining.extend(np.argwhere(np.array(flat['key'], dtype=str) == str(tt)).ravel())

            ae = Network().add_input_layer(92, normalization=Normalization.NONE)\
                .add_dense_layer(46, activation=tf.nn.tanh)\
                .add_inverse_layer(layer_index=-1, activation=tf.nn.sigmoid)
            ae.set_default_cost_method(Cost.L2_NORM)

            ae.train(x=flat['x'][aetraining], y=flat['y'][aetraining], step=0.01,max_epochs=2,threshold=0.0001,batch=2)

            net = Network().add_input_layer_from_network(ae, ae.get_deepest_hidden_layer_index())\
                .add_rnn_layer(46, activation=tf.nn.relu)\
                .begin_multi_output([Cost.RMSE, Cost.CROSS_ENTROPY])\
                .add_dense_layer(1, activation=tf.nn.sigmoid) \
                .end_multi_output()
        # endif
        else:
            # net = Network().add_input_layer(92, normalization=Normalization.Z_SCORE) \
            #     .add_lstm_layer(200, activation=tf.identity) \
            #     .add_dense_layer(1, activation=tf.nn.sigmoid)
            net = Network().add_input_layer(92, normalization=Normalization.Z_SCORE) \
                .add_lstm_layer(200, activation=tf.identity) \
                .add_dropout_layer(1,keep=0.6,activation=tf.nn.sigmoid)

        net.set_default_cost_method(Cost.CROSS_ENTROPY)

        net.train(x=seq['x'][training], y=seq['y'][training], step=0.01,
                max_epochs=2, threshold=0.0001, batch=1)

        pred = net.predict(x=seq['x'][test_set])

        fold_auc.append(Aprime(actual=my4dto2d(seq['y'][test_set]), predicted=flatten_sequence(pred[0]).ravel()))
        fold_rmse.append(eu.rmse(actual=my4dto2d(seq['y'][test_set]), predicted=flatten_sequence(pred[0]).ravel()))

        print(fold_auc[-1])
        print(fold_rmse[-1])

    print("{:=<40}".format(''))
    for i in range(len(fold_auc)):
        print("Fold {} AUC: {:<.3f}".format(i+1, fold_auc[i]))

    for i in range(len(fold_rmse)):
        print("Fold {} AUC: {:<.3f}".format(i + 1, fold_rmse[i]))

    print("{:=<40}".format(''))
    print("Average AUC: {:<.3f} ({:<.3f})".format(np.mean(fold_auc), np.std(fold_auc)))
    print("{:=<40}\n".format(''))

    print("{:=<40}".format(''))
    print("Average RMSE: {:<.3f} ({:<.3f})".format(np.mean(fold_rmse), np.std(fold_rmse)))
    print("{:=<40}\n".format(''))


def loadAndReshape(lb):
    label = 96
    if lb == 'npc':
        label = 96
    elif lb == 'fa':
        label = 97
    elif lb == 'ws':
        label = 100
    elif lb == 'rc':
        label = 99
    else:
        print('Wrong label.')
        exit(1)
    data, labels = du.read_csv('labeled_compressed92features.csv')
    du.print_descriptives(data,labels)

    seq = format_data(data, 1, [label],list(range(4,96)),3, True)

    np.save('seq_k_' + lb + '.npy', seq['key'])
    np.save('seq_x_' + lb + '.npy', seq['x'])
    np.save('seq_y_' + lb + '.npy', seq['y'])

    flat = format_data(data, 1, list(range(4,96)),list(range(4,96)),3, False)
    np.save('flat_k_' + lb + '.npy', flat['key'])
    np.save('flat_x_' + lb + '.npy', flat['x'])
    np.save('flat_y_' + lb + '.npy', flat['y'])


if __name__ == "__main__":
    # TODO: remove utility functions from here (file loading, etc) and redirect tests to use  datautility
    # 2017/10/20
    starttime = time.time()
    lb = 'ws'
    from datetime import datetime
    print(str(datetime.now()))
    #loadAndReshape(lb)
    run_npc_test(lb)
    print(str(datetime.now()))
    endtime = time.time()
    print('Time cost: ',endtime-starttime)
