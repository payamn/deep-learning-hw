
"""
I tested lots of network architectures I put all the network I tested and accuracy that I got in the Report.xlsx.
Also, all the code for each network is here. I tested on 7 networks. You can change MYnetSkip to the
network you like to test and got the accuracy. I got 74.2% accuracy using MynetSkip which is as follow:
In this network I have some skip to improve the accuracy at the starting point I will have two conv layer and then one
concat layer which will connect the input and output of this two layer so it is one the skip layer then one max pool layer
and then I have two other conv layer and another concat layer and a normalize layer then another maxpool then dropout and
one fully connected layer then another dropout and a flatten layer and finally another fully connected layer to connect
them to output classes.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import pickle
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

# address of restoring parameter
WORK_DIRECTORY = '/home/payam/PycharmProjects/Deep learning/Homework2/dipendra/data2'
IMAGE_SIZE = 32
NUM_CHANNELS = 3 #RGB
NUM_LABELS = 10 #num classes
max_accuracy = 10 #max accuracy until now
BATCH_SIZE = 100
NUM_EPOCHS = 10000
EVAL_BATCH_SIZE = 1000
EVAL_FREQUENCY = 5
SEED = None # set to None for random seed


slim = tf.contrib.slim

def cifarnet(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet'):
  """Creates a variant of the CifarNet model.
  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)
  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}

  with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0006)):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1')
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    end_points['pool2'] = net
    net = slim.flatten(net)
    end_points['Flatten'] = net
    net = slim.fully_connected(net, 384, scope='fc3')
    end_points['fc3'] = net
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    net = slim.fully_connected(net, 192, scope='fc4')
    end_points['fc4'] = net
    logits = slim.layers.fully_connected(net, NUM_LABELS, activation_fn=None, scope='fc5')

    # end_points['Logits'] = logits
    # end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits


def myCifarnet(images, is_training=True,
             dropout_keep_prob=0.5):

  end_points = {}

  with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0006)):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    net = slim.conv2d(net, 32, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    end_points['pool2'] = net
    net = slim.flatten(net)
    end_points['Flatten'] = net
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    net = slim.fully_connected(net, 192, scope='fc4')
    end_points['fc4'] = net
    logits = slim.layers.fully_connected(net, NUM_LABELS, activation_fn=None, scope='fc5')


  return logits

def SimpleModel(data, train=False):
    if train:
        reuse = None
    else:
        reuse = True
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0006)):
        net = slim.layers.conv2d(data, 64, [5, 5], 1,scope='conv1', reuse=reuse)
        net = slim.layers.max_pool2d(net, [2,2], scope='pool1')
        net = slim.layers.conv2d(net, 256, [5, 5],scope='conv2', reuse=reuse)
        net = slim.layers.max_pool2d(net, [2,2], scope='pool2')

    net = slim.layers.flatten(net, scope='flatten3')
    net = slim.layers.fully_connected(net, 2048, scope='fc1', reuse=reuse)
    if train:
        net = tf.nn.dropout(net, 0.5, seed=SEED)
    net = slim.layers.fully_connected(net, NUM_LABELS, activation_fn=None, scope='fc2', reuse=reuse)
    return net


def vgg_16(inputs,
           num_classes=10,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      # net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      # net = slim.max_pool2d(net, [2, 2], scope='pool3')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      # net = slim.max_pool2d(net, [2, 2], scope='pool4')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      # net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # # Use conv2d instead of fully_connected layers.
      # net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.layers.flatten(net, scope='flatten3')
      net = slim.layers.fully_connected(net, NUM_LABELS, activation_fn=None, scope='fc2')


      return net

def Vgg16Change(inputs,
           num_classes=10,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='Vgg16Change'):

  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(0.0006)):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      # net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      # net = slim.max_pool2d(net, [2, 2], scope='pool3')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      # net = slim.max_pool2d(net, [2, 2], scope='pool4')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      # net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # # Use conv2d instead of fully_connected layers.
      # net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 128, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.layers.flatten(net, scope='flatten3')
      net = slim.layers.fully_connected(net, NUM_LABELS, activation_fn=None, scope='fc2')


      return net

def MYnetSkip(inputs,
          is_training,
          num_classes=10,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='MYnet'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0006)):
            net1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            concat1 = tf.concat(axis=3, values=[inputs,net1])
            concat1 = slim.max_pool2d(concat1,  [2, 2], scope='pool2')

            net2 = slim.repeat(concat1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # net2 = slim.max_pool2d(net2, [2, 2], scope='pool2')
            concat2 = tf.concat(axis=3, values=[concat1,net2])
            concat2 = tf.nn.lrn(concat2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
            concat2 = slim.max_pool2d(concat2,  [2, 2], scope='pool2')
            net3 = slim.repeat(concat2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
            concat3 = tf.concat(axis=3, values=[concat2,net3])
            concat3 = slim.dropout(concat3, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net4 = slim.conv2d(concat3, 128, [1, 1], scope='fc7')
            net4 = slim.dropout(net4, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net4= slim.layers.flatten(net4, scope='flatten3')
            net4 = slim.layers.fully_connected(net4, NUM_LABELS, activation_fn=None, scope='fc2')

            return net4


# Ending defining networks



# to read our dataset
def read_pickle_files(name, oneHotOut):
    f = open(name, 'rb')
    loadOut = pickle.load(f)
    f.close()
    if oneHotOut==True:
        loadOut = np.asarray(loadOut)
        out = np.zeros((np.shape(loadOut)[0], 10), dtype=np.int)
        for x in range(0, np.shape(loadOut)[0]):
            out[x, loadOut[x]] = 1
        return out
    else :
        return loadOut





train_feat = read_pickle_files("train_feat.pickle",False)
train_lab = read_pickle_files('train_lab.pickle',True)
validation_feat = read_pickle_files('validation_feat.pickle',False)
validation_lab = read_pickle_files('validation_lab.pickle', False)
test_lab = read_pickle_files('test_lab.pickle', False)
test_feat = read_pickle_files('test_feat.pickle', False)


# # read the entire cifar10 dataset I used this code and find that if I use Cifar10 dataset I could get the accuracy of 93% but as we didn't allow to use this data set I didn't use it.
# train_cifar10 = np.zeros((50000, 32, 32,3), dtype=np.int)
#
# use cifar10 dataset
# from scipy import misc
# import glob
# for i in range (1, 50001,1):
#     train_cifar10[i-1] = misc.imread("data/images/" +str(i)+".png")
# labFromCSV= np.genfromtxt('trainLabels.csv',delimiter=',',dtype=None)
# lab_cifar10 = np.zeros((50000, 10), dtype=np.int)
# for i in range (1,50001,1):
#     if (labFromCSV[i,1]=='airplane'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,0]= 1
#     elif (labFromCSV[i,1]=='automobile'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,1]= 1
#     elif (labFromCSV[i,1]=='bird'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,2]= 1
#     elif (labFromCSV[i,1]=='cat'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,3]= 1
#     elif (labFromCSV[i,1]=='deer'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,4]= 1
#     elif (labFromCSV[i,1]=='dog'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,5]= 1
#     elif (labFromCSV[i,1]=='frog'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,6]= 1
#     elif (labFromCSV[i,1]=='horse'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,7]= 1
#     elif (labFromCSV[i,1]=='ship'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,8]= 1
#     elif (labFromCSV[i,1]=='truck'):
#         lab_cifar10[int (labFromCSV[i,0]) - 1,9]= 1
#     else:
#         print ("error " + labFromCSV[i,1] + " not found in list")




num_epochs = NUM_EPOCHS
train_size = np.shape(train_feat)[0]

#Define Place holders for inputs and outputs
train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape= (BATCH_SIZE,NUM_LABELS))
eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
is_training = tf.placeholder(tf.bool)


# to change the network just change the MYnet with the network you like all the network are in the start of the file
eval_prediction = tf.nn.softmax(MYnetSkip(eval_data, is_training=False, dropout_keep_prob=1. ))
logits = MYnetSkip(train_data_node, is_training=True)
loss = slim.losses.softmax_cross_entropy(logits, train_labels_node)


batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, batch*BATCH_SIZE, train_size*5, 0.98, staircase=True,name="learning_rate")
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

train_prediction = tf.nn.softmax(logits)

def eval(data, sess):
    size = data.shape[0]
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)

    predictions = sess.run(eval_prediction, feed_dict={eval_data:data, is_training:False})
    return predictions

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions,1)== labels) / predictions.shape[0]


def saveTestPickle(sess ,name ,test_f, test_l ):
    print ("in save test")
    predictions = eval(test_f, sess)

    data = np.argmax(predictions,1)
    f = open(name + '.pickle', 'wb')
    pickle.dump(data, f)
    f.close()
    print ("saving done")




start_time = time.time()
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print('Initialized')
    variables_to_restore = slim.get_variables_to_restore(exclude=["learning_rate"])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, WORK_DIRECTORY + "/model.ckpt")
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()

        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes

    print("total number of parameters: "+ str(total_parameters))
    # saveTestPickle(sess, "test_lab", test_feat,test_lab)   #uncommenting this line will save the test_lab file with the predicted features

    # I used premutation to shuffle the dataset and choose batches from it
    permutation = np.arange(np.shape(train_lab)[0])
    np.random.shuffle(permutation)
    xReshape1 = train_feat.reshape(-1, 32, 32, 3)
    xReshape1 = xReshape1[permutation]
    yReshape1 = train_lab[permutation]

    accuracyValidation = accuracy(eval(validation_feat, sess), validation_lab)
    print('Validation accuracy: %.1f%%' % accuracyValidation)
    if (accuracyValidation > max_accuracy):
        max_accuracy = accuracyValidation
        print(max_accuracy)
        save_path = saver.save(sess,
                               WORK_DIRECTORY + "/model.ckpt")  # will save the weight variable in the work directory

    for step in xrange (num_epochs):
        for limit in xrange(int(np.shape(train_lab)[0] / BATCH_SIZE)):
            xReshape = xReshape1[0 + limit * BATCH_SIZE:BATCH_SIZE * (limit + 1), :]
            yReshape = yReshape1[0 + limit * BATCH_SIZE:BATCH_SIZE * (limit + 1), :]
            feed_dict = {train_data_node: xReshape, train_labels_node: yReshape, is_training: True}
            _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)

        if step % EVAL_FREQUENCY == 0:
            # save_path = saver.save(sess, WORK_DIRECTORY + "/model.ckpt")   # will save the weight variable in the work directory
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Epoch  %d, %.1f ms' % (step,
                                                     1000 * elapsed_time / EVAL_FREQUENCY))
            print('loss: %.3f, learning rate: %.6f' % (l, lr))
            accuracyValidation = accuracy(eval(validation_feat, sess), validation_lab)
            print('Validation accuracy: %.1f%%' % accuracyValidation)
            if (accuracyValidation > max_accuracy ):
                max_accuracy = accuracyValidation
                print (max_accuracy)
                save_path = saver.save(sess,WORK_DIRECTORY + "/model.ckpt")  # will save the weight variable in the work directory
            sys.stdout.flush()




    print('Finished!')