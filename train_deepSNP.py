#!/usr/bin/python

import numpy as np
import sys
import os
import argparse
import time
from matplotlib import pyplot
from IPython import display

sys.path.append("/usr/local")
from caffe2.python import core, utils, model_helper, net_drawer, workspace, visualize, brew
from caffe2.proto import caffe2_pb2


# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=2'])
print("Necessities imported!")

device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)

#data_folder = "/mnt/app_hdd/scratch/blhill"
root_folder = "/nfs/home/blhill/code/github/deepSNP"
data_folder = root_folder


def AddInput(model, batch_size, db, db_type):
    # load the data
    in_data, label = model.TensorProtosDBInput(
        [], ["in_data", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(in_data, "data", to=core.DataType.FLOAT)
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label


def AddNetModel(model, data):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.

    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number of output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    '''
    # Layer 1: 100 x 30 x 30
    conv1 = brew.conv(model, data, 'conv1', dim_in=7, dim_out=100, kernel=1, use_cudnn=True)
    conv1 = brew.relu(model, conv1, conv1, use_cudnn=True)
    # Layer 2: 98 x 28 x 20
    conv2 = brew.conv(model, conv1, 'conv2', dim_in=100, dim_out=80, kernel=3, use_cudnn=True)
    conv2 = brew.relu(model, conv2, conv2, use_cudnn=True)
    #  Layer 3: 96 x 26 x 20 
    conv3 = brew.conv(model, conv2, 'conv3', dim_in=80, dim_out=60, kernel=5, use_cudnn=True)
    conv3 = brew.relu(model, conv3, conv3, use_cudnn=True)
    #  Layer 4: 94 x 24 x 20 
    conv4 = brew.conv(model, conv3, 'conv4', dim_in=60, dim_out=40, kernel=3, use_cudnn=True)
    conv4 = brew.relu(model, conv4, conv4, use_cudnn=True)
    #  Layer 5: 92 x 22 x 20 
    conv5 = brew.conv(model, conv4, 'conv5', dim_in=40, dim_out=30, kernel=3, use_cudnn=True)
    conv5 = brew.relu(model, conv5, conv5, use_cudnn=True)

    fc3 = brew.fc(model, conv5, 'fc3', dim_in=30 * 90 * 20, dim_out=1000, use_cudnn=True)
    fc3 = brew.relu(model, fc3, fc3, use_cudnn=True)

    pred = brew.fc(model, fc3, 'pred', 1000, 2, use_cudnn=True)
    softmax = brew.softmax(model, pred, 'softmax', use_cudnn=True)
    return softmax


def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = brew.iter(model, "iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)


def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
        # Now, if we really want to be verbose, we can summarize EVERY blob
        # that the model produces; it is probably not a good idea, because that
        # is going to take time - summarization do not come for free. For this
        # demo, we will only show how to summarize the parameters and their
        # gradients.

print "Starting..."
with core.DeviceScope(device_opt):
    arg_scope = {"order": "NCHW", "use_cudnn": True, "cudnn_exhaustice_search": True}
    train_model = model_helper.ModelHelper(name="deepSNP_train", arg_scope=arg_scope)
    data, label = AddInput(
        train_model, batch_size=256,
        db=os.path.join(data_folder, 'NA12878.train_400k_100W.minidb'),
        db_type='minidb')
    softmax = AddNetModel(train_model, data)
    AddTrainingOperators(train_model, softmax, label)
    AddBookkeepingOperators(train_model)
    
    # Testing model. We will set the batch size to 100, so that the testing
    # pass is 100 iterations (10,000 images in total).
    # For the testing model, we need the data input part, the main LeNetModel
    # part, and an accuracy part. Note that init_params is set False because
    # we will be using the parameters obtained from the train model.
    test_model = model_helper.ModelHelper(
        name="deepSNP_test", arg_scope=arg_scope, init_params=False)
    test_data, test_label = AddInput(
        test_model, batch_size=100,
        db=os.path.join(data_folder, 'NA12878.train_400k_100W.minidb'),
        db_type='minidb')
    test_softmax = AddNetModel(test_model, test_data)
    AddAccuracy(test_model, test_softmax, test_label)
    
    # Deployment model. We simply need the main LeNetModel part.
    deploy_model = model_helper.ModelHelper(
        name="deepSNP_deploy", arg_scope=arg_scope, init_params=False)
    AddNetModel(deploy_model, "data")
    # You may wonder what happens with the param_init_net part of the deploy_model.
    # No, we will not use them, since during deployment time we will not randomly
    # initialize the parameters, but load the parameters from the db.

with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: " + root_folder)

graph  = net_drawer.GetPydotGraph(train_model.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)

# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net, overwrite=True)
# set the number of iterations and track the accuracy & loss
total_iters = 2000000
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
total_num_ones = 0
# Now, we will manually run the network for 200 iterations.
for i in range(total_iters):
    workspace.RunNet(train_model.net)
    total_num_ones += np.sum(np.array(workspace.FetchBlob('label')))
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    if i % 5000 == 0:
        print "train iter", i, " accuracy:", accuracy[i]
print "Total number of positive training examples:", total_num_ones
# After the execution is done, let's plot the values.
fig = pyplot.figure()
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
fig.savefig("train.png")
#pyplot.show()

test_iters = 5000
# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
test_accuracy = np.zeros(test_iters)
total_num_ones = 0
for i in range(test_iters):
    workspace.RunNet(test_model.net.Proto().name)
    total_num_ones += np.sum(np.array(workspace.FetchBlob('label')))
    test_accuracy[i] = workspace.FetchBlob('accuracy')
    if i % 1000 == 0:
        print "test iter", i, " accuracy:", accuracy[i]
print "Total number of positive testing examples:", total_num_ones
print('test_accuracy: %f' % test_accuracy.mean())
# After the execution is done, let's plot the values.
fig2 = pyplot.figure()
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
fig2.savefig("test.png")
#pyplot.show()

