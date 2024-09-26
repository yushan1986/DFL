import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import random
import copy

from time import time

from CFLclasses import FrameworkCFL
from CFLclasses import BenUserCFL
from CFLclasses import MalUserCFL
from CFLclasses import ServerCFL
from CFLclasses import PlotterCFL

starttime = time()

seed = random.randint(0, 100)
seed = 69
print('Seed = ', seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

'''DATA DISTRIBUTION'''
data_dist = 'IID'  # can be RND, IID or BIA
alpha = 100
least_samples = 100

'''EXPERIMENT PARAMETERS'''
config_text = 'CFL_WEIGHT_L2_MNIST_'  # print the conf in the output
dynamic = 0
input_features = 784
num_classes = 10

defense_flag = 0

'''LEARNING PARAMETERS'''
n_epochs = 20
#mb_size = 64
lr = 0.01

'''POISONING PARAMETERS'''
poi_user_perc = 0
fraud_type = 'R'  # can be R, C or D
poison_flag = 'N'  # can be N or D or M
atk_type = 'FLIP'  # can be FLIP, MODEL_REP, GAUSS

if poison_flag == 'D':
    if atk_type == 'FLIP':
        source = 1
        target = 7
        multiple = 0  # can be 1 or 0
        if multiple == 1:
            source2 = 5
            target2 = 8

if poison_flag == 'M':
    if atk_type == 'MODEL_REP':
        source = 1
        target = 7
    if atk_type == 'GAUSS':
        mean = 0
        std_dev = 0.1

'''DEFENSE PARAMETERS'''
defense_alg = 'FEDAVG'  # can be FEDAVG or BALANCE or SHIELD
if defense_alg == 'BALANCE':
    # for BALANCE DEFENSE
    gamma = 0.3
    kk = 1
    alpha = 0.5

print('ATTACK: ', atk_type, ' and DEFENSE: ', defense_alg)

'''DEFINE NETWORK ARCHITECTURE for DFL'''
no_users = 20

cfl_framework = FrameworkCFL(no_users, alpha, least_samples, input_features, num_classes)

server = ServerCFL(input_features, num_classes, lr, n_epochs)

ben_user_ids = list(range(int((1 - poi_user_perc) * no_users)))
# ben_user_ids = [0, 1, 2, 4, 5, 7, 9, 10, 17, 19] # not in [3, 6, 8, 11, 12, 13, 14, 15, 16, 18]
ben_user_cnt = len(ben_user_ids)
mal_user_ids = list(np.setdiff1d(list(range(no_users)), ben_user_ids))
mal_user_cnt = len(mal_user_ids)

ben_users = [BenUserCFL(i, 0, input_features, num_classes, lr, n_epochs) for i in ben_user_ids]
mal_users = [MalUserCFL(i, 1, input_features, num_classes, lr, n_epochs) for i in mal_user_ids]

users = ben_users + mal_users

# load the training and test data from the files
x_train = np.load('../data/x_train.npy')
y_train = np.load('../data/y_train.npy')
x_test = np.load('../data/x_test.npy')
y_test = np.load('../data/y_test.npy')

train_samples = x_train.shape[0]
test_samples = x_test.shape[0]

# ready the test data for inference
x_test_inf = copy.deepcopy(x_test)
y_test_inf = copy.deepcopy(y_test)
x_test_inf = np.reshape(x_test_inf, (x_test_inf.shape[0], -1)) / 255

# making the feature vector between 0 and 1
x_train = np.reshape(x_train, (x_train.shape[0], -1)) / 255
x_test = np.reshape(x_test, (x_test.shape[0], -1)) / 255

# tran_smpl = 600 * no_users # 59904 # 234 * no_users # 59904
# test_smpl = 100 * no_users #9984  39 * no_users #9984
tran_smpl = 3000 * no_users
test_smpl = 500 * no_users

# extract a subset of data to
x_train = x_train[:tran_smpl, :]
y_train = y_train[0:tran_smpl]
x_test = x_test[:test_smpl, :]
y_test = y_test[0:test_smpl]

y_train = np.reshape(y_train, (y_train.shape[0], -1))
y_test = np.reshape(y_test, (y_test.shape[0], -1))

train = np.concatenate((x_train, y_train), axis=1)

train_data = pd.DataFrame(train)

x_train_k, y_train_k, count_ones = cfl_framework.create_user_data(data_dist, train_data)

'''Make Test Data'''
test_batched = tf.data.Dataset.from_tensor_slices((x_test_inf, y_test_inf)).batch(len(y_test_inf))
server.get_test_data(test_batched)

# assign data to the users
for i in range(no_users):
    users[i].get_data(x_train_k[i], y_train_k[i])


'''Poisoning by MAL users'''
if poison_flag == 'D':
    # label flipping attack
    if atk_type == 'FLIP':
        for i in mal_users:
            i.label_flip(source, target)
            if multiple == 1:
                i.label_flip(source2, target2)
    elif atk_type == 'TRIG':
        print('TRIG')
elif poison_flag == 'M':
    print('A Model Poisoning Scenario')

'''To store outputs'''
total_loss = np.zeros([n_epochs, 1])
acc_test = np.zeros([n_epochs, 1])

acc_test_cls1 = np.zeros([n_epochs, 1])
acc_test_cls7 = np.zeros([n_epochs, 1])
acc_test_cls5 = np.zeros([n_epochs, 1])
acc_test_cls8 = np.zeros([n_epochs, 1])

print('time for data proc: ', time()-starttime)


for comm_round in range(n_epochs):
    print('iteration: ', comm_round)
    print(tf.reduce_sum(server.model.weights[0]))
    print(tf.reduce_sum(server.model.weights[1]))

    '''SET INITIAL WEIGHTS'''
    # get the model weights from the server for the first round
    for i in users:
        i.model.set_weights(server.model.get_weights())

    # Simulate each client's local training
    for i in users:
        i.train()

    client_weights = []
    for i in users:
        weights = i.send_weights()
        client_weights.append(weights)

    # Average model weights from all clients
    server.fedavg_user_models(client_weights)
    print(tf.reduce_sum(server.model.weights[0]))
    print(tf.reduce_sum(server.model.weights[1]))
    server.test_model(comm_round, num_classes)

    '''EXTRACT PERFORMANCE METRICS FROM SERVER'''

    acc_test[comm_round] = server.testacc
    total_loss[comm_round] = server.testloss
    server.calc_class_wise_acc(num_classes)
    acc_test_cls1[comm_round] = server.class_wise_acc[1]
    acc_test_cls7[comm_round] = server.class_wise_acc[7]
    acc_test_cls5[comm_round] = server.class_wise_acc[5]
    acc_test_cls8[comm_round] = server.class_wise_acc[8]

'''PLOTS'''
plotter = PlotterCFL(n_epochs, acc_test, total_loss, acc_test_cls1, acc_test_cls7, acc_test_cls5, acc_test_cls8)

plotter.plot_acc()
plotter.plot_loss()

print('total time: ', time()-starttime)
