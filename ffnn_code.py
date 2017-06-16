import tensorflow as tf
from numpy import sqrt, repeat, average
import numpy as np
from numpy.lib.npyio import genfromtxt
from math import ceil, floor, inf
from random import sample
from scipy.stats import mode
from sklearn.metrics import roc_auc_score

def load_data(images_filename, labels_filename):
    images_temp = np.array(genfromtxt(images_filename, dtype=int, delimiter=',', names=True).tolist())
    labels_temp = np.array(genfromtxt(labels_filename, dtype=int, delimiter=',', names=True).tolist())
    
    images = images_temp.reshape([-1,150])
    labels = labels_temp.reshape([-1,1])
    
    #make features
    sum_wn = np.sum(images, 1)
    sum_abs = np.sum(np.abs(images), 1)
    count_p = np.sum(np.ceil((images+1)/2), 1)
    percent_correct = np.atleast_2d((sum_wn + sum_abs)/(2*count_p)).reshape([-1,1])
    
    return percent_correct, images, labels

def run_majority_class():
    _, _, training_labels = load_data("training_images.csv", "training_labels.csv")
    _, _, testing_labels = load_data("testing_images.csv", "testing_labels.csv")
    
    majority_class = mode(training_labels)[0]
    predictions = repeat(majority_class, testing_labels.shape[0])
    prediction_accuracy = average(predictions == testing_labels)
    prediction_auc = roc_auc_score(testing_labels, predictions)
    return prediction_accuracy, prediction_auc

def one_layer_nn(hiddenLayerSize, learnRate, maxEpochs, minibatchSize, printLast20, printEpoch, printAll):
    _, training_images, training_labels = load_data("training_images.csv", "training_labels.csv")
    _, testing_images, testing_labels = load_data("testing_images.csv", "testing_labels.csv")
    inputDim = 150
    outputDim = 1
    
    x = tf.placeholder(tf.float32, [None, inputDim])
    y = tf.placeholder(tf.float32, [None, outputDim])
    
    w1 = tf.Variable(tf.random_normal([inputDim, hiddenLayerSize], mean = 0, stddev = 1/sqrt(inputDim)))
    wx = tf.Variable(tf.random_normal([hiddenLayerSize, outputDim], mean = 0, stddev = 1/sqrt(hiddenLayerSize)))
    b1 = tf.Variable(0.01*tf.ones([hiddenLayerSize]))
    bx = tf.Variable(0.01*tf.ones([outputDim]))
    
    z1 = tf.matmul(x, w1) + b1
    h1 = tf.nn.relu(z1)
    zx = tf.matmul(h1, wx) + bx
    hx = tf.nn.sigmoid(zx)
    
    #cross_entropy = -tf.reduce_mean((y + small_constant) * tf.log(hx + small_constant))
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=zx))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h2), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_mean(tf.abs(y - h3) + small_constant)
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = zx))
    correct_prediction = tf.equal(tf.round(hx), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(tf.reduce_mean(w1) + tf.reduce_mean(w2) + tf.reduce_mean(w3))
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    
    numRow = training_images.shape[0]
    numBatch = ceil(numRow/minibatchSize)
    subsample = np.array(sample(list(repeat(range(0,numBatch),minibatchSize)),numRow))
    for iteration in range(maxEpochs * numBatch):
        batchId = iteration%numBatch
        #print(batchId, iteration, numBatch, numRow, minibatchSize)
        batch_xs = training_images[subsample == batchId,]
        batch_ys = training_labels[subsample == batchId,]
        
        trainingBatchDict = {x: batch_xs, y: batch_ys}
        sess.run(train_step, feed_dict=trainingBatchDict)
        if(printEpoch and batchId == numBatch -1):
            print("epoch#", floor(iteration/numBatch), end = ";\t")
            print("training cost:", sess.run(cross_entropy, feed_dict=trainingBatchDict),  end = ";\t")
            print("training accuracy", sess.run(accuracy, feed_dict=trainingBatchDict))
            #print("training auc", roc_auc_score(batch_ys, sess.run(tf.round(hx), feed_dict=trainingBatchDict)))
        elif printAll or (printLast20 and (maxEpochs * numBatch) - iteration <= 20):
            print("iteration#", iteration, end = ";\t")
            print("training cost:", sess.run(cross_entropy, feed_dict=trainingBatchDict),  end = ";\t")
            print("training accuracy", sess.run(accuracy, feed_dict=trainingBatchDict))
            #print("training auc", roc_auc_score(batch_ys, sess.run(tf.round(hx), feed_dict=trainingBatchDict)))
            #print("hx accuracy", sess.run(hx, feed_dict=trainingBatchDict).reshape([1,-1]))
    
    trainingDict = {x: training_images, y: training_labels}
    training_cost     = sess.run(cross_entropy, feed_dict=trainingDict)
    training_accuracy = sess.run(accuracy,      feed_dict=trainingDict)
    training_auc      = roc_auc_score(training_labels, sess.run(tf.round(hx), feed_dict=trainingDict))
    
#     validationDict = {x: testing_images, y: testing_labels}
#     validation_cost     = sess.run(cross_entropy, feed_dict=validationDict)
#     validation_accuracy = sess.run(accuracy,      feed_dict=validationDict)
#     validation_auc      = sess.run(auc,           feed_dict=validationDict)
    
    testingDict = {x: testing_images, y: testing_labels}
    test_cost     = sess.run(cross_entropy, feed_dict=testingDict)
    test_accuracy = sess.run(accuracy,      feed_dict=testingDict)
    test_auc      = roc_auc_score(testing_labels, sess.run(tf.round(hx), feed_dict=testingDict))
    
    #print(training_cost, training_accuracy, training_auc, validation_cost, validation_accuracy, validation_auc, test_cost, test_accuracy, test_auc)
    
    #return training_cost, training_accuracy, training_auc, validation_cost, validation_accuracy, validation_auc, test_cost, test_accuracy, test_auc
    return training_cost, training_accuracy, training_auc, test_cost, test_accuracy, test_auc


def run_simple_nn(hiddenLayerSize, learnRate, maxEpochs, minibatchSize, printLast20, printEpoch, printAll):
    _, training_images, training_labels = load_data("training_images.csv", "training_labels.csv")
    _, testing_images, testing_labels = load_data("testing_images.csv", "testing_labels.csv")
    inputDim = 150
    outputDim = 1
    
    x = tf.placeholder(tf.float32, [None, inputDim])
    y = tf.placeholder(tf.float32, [None, outputDim])
    
    w1 = tf.Variable(tf.random_normal([inputDim, hiddenLayerSize[0]], mean = 0, stddev = 1/sqrt(inputDim)))
    w2 = tf.Variable(tf.random_normal([hiddenLayerSize[0], hiddenLayerSize[1]], mean = 0, stddev = 1/sqrt(hiddenLayerSize[0])))
    w3 = tf.Variable(tf.random_normal([hiddenLayerSize[1], hiddenLayerSize[2]], mean = 0, stddev = 1/sqrt(hiddenLayerSize[1])))
    wx = tf.Variable(tf.random_normal([hiddenLayerSize[2], outputDim], mean = 0, stddev = 1/sqrt(hiddenLayerSize[2])))
    b1 = tf.Variable(0.01*tf.ones([hiddenLayerSize[0]]))
    b2 = tf.Variable(0.01*tf.ones([hiddenLayerSize[1]]))
    b3 = tf.Variable(0.01*tf.ones([hiddenLayerSize[2]]))
    bx = tf.Variable(0.01*tf.ones([outputDim]))
    
    z1 = tf.matmul(x, w1) + b1
    h1 = tf.nn.relu(z1)
    z2 = tf.matmul(h1, w2) + b2
    h2 = tf.nn.relu(z2)
    z3 = tf.matmul(h2, w3) + b3
    h3 = tf.nn.relu(z3)
    zx = tf.matmul(h3, wx) + bx
    hx = tf.nn.sigmoid(zx)
    
    #cross_entropy = -tf.reduce_mean((y + small_constant) * tf.log(hx + small_constant))
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=zx))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h2), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_mean(tf.abs(y - h3) + small_constant)
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = zx))
    correct_prediction = tf.equal(tf.round(hx), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(tf.reduce_mean(w1) + tf.reduce_mean(w2) + tf.reduce_mean(w3))
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    
    numRow = training_images.shape[0]
    numBatch = ceil(numRow/minibatchSize)
    subsample = np.array(sample(list(repeat(range(0,numBatch),minibatchSize)),numRow))
    for iteration in range(maxEpochs * numBatch):
        batchId = iteration%numBatch
        #print(batchId, iteration, numBatch, numRow, minibatchSize)
        batch_xs = training_images[subsample == batchId,]
        batch_ys = training_labels[subsample == batchId,]
        
        trainingBatchDict = {x: batch_xs, y: batch_ys}
        sess.run(train_step, feed_dict=trainingBatchDict)
        if(printEpoch and batchId == numBatch -1):
            print("epoch#", floor(iteration/numBatch), end = ";\t")
            print("training cost:", sess.run(cross_entropy, feed_dict=trainingBatchDict),  end = ";\t")
            print("training accuracy", sess.run(accuracy, feed_dict=trainingBatchDict))
            #print("training auc", roc_auc_score(batch_ys, sess.run(tf.round(hx), feed_dict=trainingBatchDict)))
        elif printAll or (printLast20 and (maxEpochs * numBatch) - iteration <= 20):
            print("iteration#", iteration, end = ";\t")
            print("training cost:", sess.run(cross_entropy, feed_dict=trainingBatchDict),  end = ";\t")
            print("training accuracy", sess.run(accuracy, feed_dict=trainingBatchDict))
            #print("training auc", roc_auc_score(batch_ys, sess.run(tf.round(hx), feed_dict=trainingBatchDict)))
            #print("hx accuracy", sess.run(hx, feed_dict=trainingBatchDict).reshape([1,-1]))
    
    trainingDict = {x: training_images, y: training_labels}
    training_cost     = sess.run(cross_entropy, feed_dict=trainingDict)
    training_accuracy = sess.run(accuracy,      feed_dict=trainingDict)
    training_auc      = roc_auc_score(training_labels, sess.run(tf.round(hx), feed_dict=trainingDict))
    
#     validationDict = {x: testing_images, y: testing_labels}
#     validation_cost     = sess.run(cross_entropy, feed_dict=validationDict)
#     validation_accuracy = sess.run(accuracy,      feed_dict=validationDict)
#     validation_auc      = sess.run(auc,           feed_dict=validationDict)
    
    testingDict = {x: testing_images, y: testing_labels}
    test_cost     = sess.run(cross_entropy, feed_dict=testingDict)
    test_accuracy = sess.run(accuracy,      feed_dict=testingDict)
    test_auc      = roc_auc_score(testing_labels, sess.run(tf.round(hx), feed_dict=testingDict))
    
    #print(training_cost, training_accuracy, training_auc, validation_cost, validation_accuracy, validation_auc, test_cost, test_accuracy, test_auc)
    
    #return training_cost, training_accuracy, training_auc, validation_cost, validation_accuracy, validation_auc, test_cost, test_accuracy, test_auc
    return training_cost, training_accuracy, training_auc, test_cost, test_accuracy, test_auc
    
def find_best_parameters_for_simple_nn():
    possibleHiddenLayerSizes = [50,40,30,20]
    possibleLearnRates = [5e-1,1e-1,1e-2,1e-3]
    possibleMiniBatchSizes = [256,128,64,32]
    maxEpochs = 100
    printLast20 = False
    printEpoch = False
    printAll = False
    
    bestHL1 = possibleHiddenLayerSizes[0]
    bestHL2 = possibleHiddenLayerSizes[0]
    bestHL3 = possibleHiddenLayerSizes[0]
    bestHL4 = possibleHiddenLayerSizes[0]
    bestLR = possibleLearnRates[0]
    bestBS = possibleMiniBatchSizes[0]
    
    bestCost = inf
    
    print("currentBestSetting", [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS)
    print("currentBestCost", bestCost) 
    
    for HL1 in possibleHiddenLayerSizes:
        thisCost, _, _, _, _, _ = run_simple_nn([HL1, bestHL2, bestHL3, bestHL4], bestLR, maxEpochs, bestBS, printLast20, printEpoch, printAll)
        if thisCost < bestCost:
            bestHL1 = HL1
            bestCost = thisCost
    print("currentBestSetting", [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS)
    print("currentBestCost", bestCost) 
    
    for HL2 in possibleHiddenLayerSizes:
        thisCost, _, _, _, _, _ = run_simple_nn([bestHL1, HL2, bestHL3, bestHL4], bestLR, maxEpochs, bestBS, printLast20, printEpoch, printAll)
        if thisCost < bestCost:
            bestHL2 = HL2
            bestCost = thisCost
    print("currentBestSetting", [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS)
    print("currentBestCost", bestCost) 
            
    for HL3 in possibleHiddenLayerSizes:
        thisCost, _, _, _, _, _ = run_simple_nn([bestHL1, bestHL2, HL3, bestHL4], bestLR, maxEpochs, bestBS, printLast20, printEpoch, printAll)
        if thisCost < bestCost:
            bestHL3 = HL3
            bestCost = thisCost
    print("currentBestSetting", [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS)
    print("currentBestCost", bestCost) 
            
    for HL4 in possibleHiddenLayerSizes:
        thisCost, _, _, _, _, _ = run_simple_nn([bestHL1, bestHL2, bestHL3, HL4], bestLR, maxEpochs, bestBS, printLast20, printEpoch, printAll)
        if thisCost < bestCost:
            bestHL4 = HL4
            bestCost = thisCost
    print("currentBestSetting", [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS)
    print("currentBestCost", bestCost) 
            
    for LR in possibleLearnRates:
        thisCost, _, _, _, _, _ = run_simple_nn([bestHL1, bestHL2, bestHL3, bestHL4], LR, maxEpochs, bestBS, printLast20, printEpoch, printAll)
        if thisCost < bestCost:
            bestLR = LR
            bestCost = thisCost
            
    for BS in possibleMiniBatchSizes:
        thisCost, _, _, _, _, _ = run_simple_nn([bestHL1, bestHL2, bestHL3, bestHL4], bestLR, maxEpochs, BS, printLast20, printEpoch, printAll)
        if thisCost < bestCost:
            bestBS = BS
            bestCost = thisCost
    print("currentBestSetting", [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS)
    print("currentBestCost", bestCost) 
            
    return [bestHL1, bestHL2, bestHL3, bestHL4], bestLR, bestBS
    
if __name__ == "__main__":
    #print(run_simple_nn(hiddenLayerSize, learnRate, maxEpochs, minibatchSize, printLast20, printEpoch, printAll))
    majority_class_accuracy, majority_class_auc = run_majority_class()
    print("majority class: accuracy =",majority_class_accuracy,"auc =", majority_class_auc)
    _, _, _, test_cost_1l, test_accuracy_1l, test_auc_1l = one_layer_nn(hiddenLayerSize = 50, learnRate = 0.1, maxEpochs = 500, minibatchSize = 32, printLast20 = True, printEpoch = True, printAll = False)
    print("single layer NN: cost", test_cost_1l, ", accuracy =",test_accuracy_1l,"auc =", test_auc_1l)
    bestHL, bestLR, bestBS = find_best_parameters_for_simple_nn()
    print("best feed-forward NN setting: layers ", bestHL, ",learnRate", bestLR, ",mini batch size", bestBS)
    _, _, _, test_cost, test_accuracy, test_auc = run_simple_nn(hiddenLayerSize = bestHL, learnRate = bestLR, maxEpochs = 500, minibatchSize = bestBS, printLast20 = True, printEpoch = True, printAll = False)
    print("bestNN test set: cost", test_cost, ", accuracy", test_accuracy, ", test_auc", test_auc)
    
    
    
