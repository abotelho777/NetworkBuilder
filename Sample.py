import datautility as du
import numpy as np
import time
import tf_network as tfnet
from tf_network import Network, Cost, Normalization, Optimizer
import evaluationutility as eu
import tensorflow as tf


def run_sample():
    # we will evaluate with a 5 fold cross validation
    n_folds = 5

    # utility function can be used to load the csv with column headers
    data, headers = du.read_csv('resources/affect_obs/aligned_affect_observations_no_unlabeled.csv')

    """ 
    keys are used to create each sample sequence
    
    here we are creating each sequence to be defined by the distinct value pairs of the 6th and 0th column
    example: each student assignment could be defined as a separate training sample by using columns 6 and 7
    """
    key_cols = [6, 0]

    """
    define the list of columns containing the features to input into the network
    """
    cov = list(range(9, 101))

    """ 
    labels are defined as a list of lists
    - each label set corresponds with a particular output that may be comprised of several classes
    example: here the model is set up to output 4 values for the first set, 1 value for the second, 
    1 for the third, etc.
    
    This allows for the prediction of multiple outcomes (i.e. completion AND student affect AND ...)
    """
    label_set1 = [101, 102, 103, 104]
    label_set2 = [101]
    label_set3 = [102]
    label_set4 = [103]
    label_set5 = [104]

    """
    define the column on which to sort values within each sequence
    """
    sortby = 1

    # utility function to print the column indices, variable names, and some descriptive stats about the data
    du.print_descriptives(data,headers)

    """
    The format_data function is needed to generate data in the proper format for the network.
    This IS NEEDED regardless of the structure of the data, and puts together each of the variables
    defined above
    
    The last parameter defines whether the data is a sequence/time series
    
    The output of the function is a dictionary object with three keys: 'x', 'y', and 'key'
    """
    seq = tfnet.format_data(data, key_cols, [label_set1, label_set2, label_set3, label_set4, label_set5], cov, sortby,
                            True)

    # we can then find the average length of the produced sequences (this will be 1 for non-sequential data
    print('Average Sequence Length: {}'.format(np.mean([len(i) for i in seq['x']])))

    """
    fold by the first (0) column of the key (user_id) - this is stored as an additional key value at index 0
    The second parameter can be a single value or a list of key column indicies (here we are folding 
    by student, even if our samples are at a sub-student level
    """
    seq['key'] = tfnet.fold_by_key(seq['key'], 0, n_folds)

    # each part of the dictionary object is a numpy array and can be saved as such if reformatting takes too long
    np.save('seq_k_sample.npy', seq['key'])
    np.save('seq_x_sample.npy', seq['x'])
    np.save('seq_y_sample.npy', seq['y'])



    # once saved, we can ignore everything up
    seq = dict()
    seq['x'] = np.load('seq_x_sample.npy')
    seq['y'] = np.load('seq_y_sample.npy')
    seq['key'] = np.load('seq_k_sample.npy')


    """
    The seq['y'] is a 4-dimensional array to support multiple label sets
    As such, utility functions can be used to help manipulate these values:
    

    fill_input_multi_label(sequence_y, sequence_x, network)
    -- used for auto-encoders, will replace placeholder values in the label set with the 
    -- output of the input layer of the network
    
    extract_from_multi_label(sequence_y, labels)
    -- returns a 4D array by extracting a set of label sets (labels can be a single index or array of indices)
    
    reverse_multi_label(sequence_y, labels)
    -- reverses the sequence of labels within each label set sequence (labels can be a single index or array of indices)
    
    merge_multi_label(sequence_y1, sequence_y2=None)
    -- merges two 4D 'y' arrays into a single 4D 'y' array
    
    one_hot_multi_label(sequence_y, labels)
    -- generates a one-hot encoding of each label set defined by 'labels' (must be single-value label sets)
    
    ravel_multi_label(sequence_y)
    -- merges all label sets into a series of single-valued label sets (e.g. a label set of 4 classes and 1 class 
    -- will become 5 label sets of each 1 value)
    
    find_in_multi_label(sequence_y, value)
    -- searches the 'y' array for a value and returns a list of 4D indices where the value was found
    
    find_and_replace_in_multi_label(sequence_y, find_value, replace_value, replace_all_classes=False)
    -- searches and replaces a value within the 'y' array
    -- the last parameter allows for the replacement of all classes if a value is found (i.e. replace all 
    -- classes with np.nan if one of the classes is -1)
    
    replace_in_multi_label(sequence_y, indices, replace_value)
    -- given a list of 4D indices, replace the value in these locations (works with find_in_multi_label)
    
    use_last_multi_label(sequence_y, labels)
    -- keeps only the last label in each sequence and sets all others to np.nan
    
    offset_label_timestep(y, label=0)
    -- offset each label by one time step in each sequence (to have the model predict a value of the next time step)
    
    describe_multi_label(sequence_y, print_description=False, print_descriptives=False)
    -- returns (and can print) a description of the format of a 'y' array (how many label sets, 
    -- how many classes per label set)
    
    
    example below:
    """
    # we can extract the first label set (with 4 classes)...
    first_label_set = tfnet.extract_from_multi_label(seq['y'], 0)

    # ravel that to become 4 label sets (each with 1 class)...
    raveled_label_set = tfnet.ravel_multi_label(first_label_set)

    # and merge the first with the raveled to get the same y array
    redundant_y = tfnet.merge_multi_label(first_label_set, raveled_label_set)

    # redundant_y and seq['y'] are identical due to how we defined the label sets when formatting
    tfnet.describe_multi_label(redundant_y, True)
    tfnet.describe_multi_label(seq['y'], True)
    ###
    # end of example
    ###

    # we can get the number of input nodes by looking at our formatted data
    n_cov = len(seq['x'][0][0])

    desc = tfnet.describe_multi_label(seq['y'], True)

    # now for model training - we use a for loop for the cross validation
    for i in range(n_folds):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        np.random.seed(1)

        """
        BEGIN MODEL BUILDING
        """
        # split the data into a training and test set using the already-folded key values
        training = np.argwhere(np.array(seq['key'][:, 0], dtype=np.int32) != i).ravel()
        test_set = np.argwhere(np.array(seq['key'][:, 0], dtype=np.int32) == i).ravel()

        # then we build the network
        net = Network().add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
        net.add_lstm_layer(100, activation=tf.identity)

        # layers between the begin/end multi output functions are created at the same level
        # and correspond with each of the label sets we've defined
        net.begin_multi_output(cost_methods=[Cost.MULTICLASS_CROSS_ENTROPY,
                                             Cost.BINARY_CROSS_ENTROPY,
                                             Cost.BINARY_CROSS_ENTROPY,
                                             Cost.BINARY_CROSS_ENTROPY,
                                             Cost.BINARY_CROSS_ENTROPY])
        net.add_dropout_layer(4, keep=0.5, activation=tf.nn.softmax)
        net.add_dropout_layer(1, keep=0.5, activation=tf.nn.sigmoid)
        net.add_dropout_layer(1, keep=0.5, activation=tf.nn.sigmoid)
        net.add_dropout_layer(1, keep=0.5, activation=tf.nn.sigmoid)
        net.add_dropout_layer(1, keep=0.5, activation=tf.nn.sigmoid)
        net.end_multi_output()

        """
        END MODEL BUILDING
        """

        # default cost method is used if not defined in the begin_multi_output
        net.set_default_cost_method(Cost.RMSE)

        # the optimizer defines how training updates are applied
        net.set_optimizer(Optimizer.ADAM)

        """
        BEGIN MODEL TRAINING
        
        -- this function will train on the supplied data until either it hits the max epochs
        -- or if a delta threshold is reached (uses a validation set and a 5-epoch moving average)
        -- Here, if the difference between performance on the validation set drops below 0 (i.e. the model 
        -- starts getting worse), it will stop
        
        -- step size defines how much to update the model weights after each batch
        """
        net.train(x=seq['x'][training],
                  y=seq['y'][training],
                  step=1e-3,
                  use_validation=True,
                  max_epochs=100, threshold=0.0, batch=3)
        """
        END MODEL TRAINING
        """

        """ 
        BEGIN MODEL TESTING
        
        -- the predict function returns a list of prediction values, each element of the list being a 3D array
        -- The length of the returned list corresponds to the number of label sets
        """
        pred = net.predict(x=seq['x'][test_set])

        # loop through each label set...
        for p in range(len(pred)):
            # flatten predictions and labels (from 3D or 4D array) back into a 2D array
            # *also useful for writing predictions/labels to a csv
            predicted = tfnet.flatten_sequence(pred[p])
            actual = tfnet.flatten_sequence(tfnet.extract_from_multi_label(seq['y'][test_set], p))

            # evaluate the model using various metrics (such as AUC and kappa)
            auc = eu.auc(actual, predicted, average_over_labels=True)
            print("Fold AUC (Label Set {}): {:<.3f}".format(p, auc))

            if desc['n_labels'][p] > 1:
                kpa = eu.cohen_kappa_multiclass(actual, predicted)
            else:
                kpa = eu.cohen_kappa(actual, predicted, average_over_labels=True)

            print("Fold KAPPA (Label Set {}): {:<.3f}".format(p, kpa))

            print("")


if __name__ == "__main__":
    run_sample()


