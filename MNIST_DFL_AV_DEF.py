import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import (array, dot, arccos, clip)
import random
import re


# import tensorflow_addons as tfa

tf.compat.v1.enable_eager_execution()
import os
import copy
import math
import hdbscan
from time import time
#from operator import itemgetter
from sklearn.metrics.pairwise import pairwise_distances

# load the classes
from classes import User

# load user defined functions
from func import counts_from_cum
from func import mini_batches
from func import mini_batches_ind
from func import mini_batches_ind_allflip
from func import accuracy
from func import get_gradients
from func import label_flip_multiple
from func import label_flip_multiple_ind
from func import make_random_data
from func import make_iid_data
from func import make_biased_data
from func import agg_device_layer
from func import agg_int_layer
from func import agg_top_layer
from func import agg_int_layer1
from func import agg_top_layer1
from func import novel_defense_device_layer
from func import novel_defense_top_layer
from func import novel_defense_int_layer

seed = random.randint(0, 100)
print(seed)
#seed = 8
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

'''DATA DISTRIBUTION'''
data_dist = 'IID'  # can be RND, IID or BIA
if data_dist == 'BIA':
    alpha = 1
    least_samples = 100

'''EXPERIMENT PARAMETERS'''
config_text = 'MNIST_DFL_AV'  # print the conf in the output
dynamic = 0

'''LEARNING PARAMETERS'''
n_epochs = 200
mb_size = 64
lr = 0.001

'''POISONING PARAMETERS'''
poisoned_flag = 1  # can be 1 or 0
if poisoned_flag == 0:
    poi_user_perc = 0
    fraud_type = 'N'
else:
    poi_user_perc = 0.40
    fraud_type = 'R'  # can be R, C or D
    multiple = 0  # can be 1 or 0
defense_flag = 1  # can be 1 or 0

'''DEFINE NETWORK ARCHITECTURE for DFL'''
no_users = 50
df = pd.read_csv('data/finalgraph10.csv', sep=';', header=None, keep_default_na=False)
df = df.iloc[0:n_epochs*no_users, :]
graph = []
for i in range(1, n_epochs+1):
    sub_df = df[df[0] == i]
    sub_graph = np.empty([no_users, no_users], dtype=int)
    for j in range(no_users):
        node = sub_df.loc[[j+(i-1)*no_users]][1][j+(i-1)*no_users]
        a = sub_df.loc[[j+(i-1)*no_users]][2][j+(i-1)*no_users]
        temp = re.findall(r'\d+', a)
        neighbours = list(map(int, temp))
        temp_vec = np.zeros(no_users)
        np.put(temp_vec, neighbours, 1)
        sub_graph[node, :] = temp_vec
    np.fill_diagonal(sub_graph, 1)
    graph.append(sub_graph)

# load the training and test data from the files
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

train_samples = x_train.shape[0]
test_samples = x_test.shape[0]
input_features = 784
num_classes = 10

# ready the test data for inference
x_test_inf = copy.deepcopy(x_test)
y_test_inf = copy.deepcopy(y_test)
x_test_inf = np.reshape(x_test_inf, (x_test_inf.shape[0], -1)) / 255
y_test_inf = tf.keras.utils.to_categorical(y_test_inf, num_classes)
x_test_inf = tf.convert_to_tensor(x_test_inf, dtype=np.float32)
y_test_inf = tf.convert_to_tensor(y_test_inf, dtype=np.float32)

# making the feature vector between 0 and 1
x_train = np.reshape(x_train, (x_train.shape[0], -1)) / 255
x_test = np.reshape(x_test, (x_test.shape[0], -1)) / 255

#tran_smpl = 600 * no_users # 59904 # 234 * no_users # 59904
#test_smpl = 100 * no_users #9984  39 * no_users #9984
tran_smpl = 1200 * no_users
test_smpl = 200 * no_users

# extract a subset of data to
x_train = x_train[:tran_smpl, :]
y_train = y_train[0:tran_smpl]
x_test = x_test[:test_smpl, :]
y_test = y_test[0:test_smpl]

y_train = np.reshape(y_train, (y_train.shape[0], -1))
y_test = np.reshape(y_test, (y_test.shape[0], -1))

train = np.concatenate((x_train, y_train), axis=1)
test = np.concatenate((x_test, y_test), axis=1)

train_data = pd.DataFrame(train)
test_data = pd.DataFrame(test)

# Separate the training and test data into classes. Output is a list of numpy arrays for both train and test
train_ordered = [[] for i in range(num_classes)]
test_ordered = [[] for i in range(num_classes)]

for i in range(num_classes):
    train_ordered[i] = train_data[train_data[784] == i]
    test_ordered[i] = test_data[test_data[784] == i]

# we observe only these five classes and their behaviors.
x_test_cls0 = (test_ordered[0].iloc[:, :-1]).to_numpy()
x_test_cls0 = tf.convert_to_tensor(x_test_cls0, dtype=np.float32)
y_test_cls0 = (test_ordered[0].iloc[: , -1]).to_numpy()
y_test_cls0 = tf.keras.utils.to_categorical(y_test_cls0, num_classes)
y_test_cls0 = tf.convert_to_tensor(y_test_cls0, dtype=np.float32)

x_test_cls1 = (test_ordered[1].iloc[:, :-1]).to_numpy()
x_test_cls1 = tf.convert_to_tensor(x_test_cls1, dtype=np.float32)
y_test_cls1 = (test_ordered[1].iloc[: , -1]).to_numpy()
y_test_cls1 = tf.keras.utils.to_categorical(y_test_cls1, num_classes)
y_test_cls1 = tf.convert_to_tensor(y_test_cls1, dtype=np.float32)

x_test_cls2 = (test_ordered[2].iloc[:, :-1]).to_numpy()
x_test_cls2 = tf.convert_to_tensor(x_test_cls2, dtype=np.float32)
y_test_cls2 = (test_ordered[2].iloc[: , -1]).to_numpy()
y_test_cls2 = tf.keras.utils.to_categorical(y_test_cls2, num_classes)
y_test_cls2 = tf.convert_to_tensor(y_test_cls2, dtype=np.float32)

x_test_cls5 = (test_ordered[5].iloc[:, :-1]).to_numpy()
x_test_cls5 = tf.convert_to_tensor(x_test_cls5, dtype=np.float32)
y_test_cls5 = (test_ordered[5].iloc[: , -1]).to_numpy()
y_test_cls5 = tf.keras.utils.to_categorical(y_test_cls5, num_classes)
y_test_cls5 = tf.convert_to_tensor(y_test_cls5, dtype=np.float32)

x_test_cls7 = (test_ordered[7].iloc[:, :-1]).to_numpy()
x_test_cls7 = tf.convert_to_tensor(x_test_cls7, dtype=np.float32)
y_test_cls7 = (test_ordered[7].iloc[: , -1]).to_numpy()
y_test_cls7 = tf.keras.utils.to_categorical(y_test_cls7, num_classes)
y_test_cls7 = tf.convert_to_tensor(y_test_cls7, dtype=np.float32)

x_test_cls8 = (test_ordered[8].iloc[:, :-1]).to_numpy()
x_test_cls8 = tf.convert_to_tensor(x_test_cls8, dtype=np.float32)
y_test_cls8 = (test_ordered[8].iloc[: , -1]).to_numpy()
y_test_cls8 = tf.keras.utils.to_categorical(y_test_cls8, num_classes)
y_test_cls8 = tf.convert_to_tensor(y_test_cls8, dtype=np.float32)

if data_dist == 'RND':
    '''Random Data Preparation'''
    x_train_rnd, y_train_rnd = make_random_data(train_data, no_users, num_classes)
    flip_data = copy.deepcopy(y_train_rnd)  # Keep the y_train as non-flipped data and flips y labels in flip_data
elif data_dist == 'IID':
    '''IID Data Preparation'''
    x_train_iid, y_train_iid = make_iid_data(train_ordered, no_users, input_features, num_classes)
    flip_data = copy.deepcopy(y_train_iid)  # Keep the y_train as non-flipped data and flips y labels in flip_data
else:
    '''Biased Data Preparation'''
    x_train_bia, y_train_bia, count_ones = make_biased_data(train_data, no_users, alpha, least_samples, num_classes)
    flip_data = copy.deepcopy(y_train_bia)

'''Poisoning user selection'''
fraud_users = []
if poisoned_flag == 1:
    if fraud_type == 'R':
        '''RANDOMLY DISTRIBUTED POISONERS'''
        fraud_users = random.sample(list(range(no_users)), int(poi_user_perc * no_users))

        #fraud_users = [1]  # manual insert
        fraud_users.sort()
        fraud_users_cnt = len(fraud_users)


    print(fraud_users)

    '''Actual Poisoning - Label Flipping'''
    if multiple == 1:
        flipped_data1, flipped_indexes1 = label_flip_multiple_ind(flip_data, fraud_users, [1] * fraud_users_cnt, [7] * fraud_users_cnt, 1)
        flipped_data2, flipped_indexes2 = label_flip_multiple_ind(flipped_data1, fraud_users, [5] * fraud_users_cnt, [8] * fraud_users_cnt, 1)

        flipped_indexes = []
        len1 = len(flipped_indexes1)
        len2 = len(flipped_indexes2)
        if len1 == len2:
            for i in range(len1):
                flipped_indexes.append([np.concatenate((flipped_indexes1[i][0], flipped_indexes2[i][0]), axis=0)])
        else:
            print('PROBLEM')
    else:
        flipped_data1, flipped_indexes = label_flip_multiple_ind(flip_data, fraud_users, [1] * fraud_users_cnt, [7] * fraud_users_cnt, 1)
        flipped_data2 = flipped_data1

    benign_users = list(np.setdiff1d(list(range(no_users)), fraud_users))
    print('POISONED', fraud_users_cnt, ' USERS')
    if data_dist == 'RND':
        x_train_k = x_train_rnd
        y_train_k = flipped_data2
    elif data_dist == 'IID':
        x_train_k = x_train_iid
        y_train_k = flipped_data2
    else:
        x_train_k = x_train_bia
        y_train_k = flipped_data2

else:
    benign_users = list(range(no_users))
    print('NOT POISONED')
    if data_dist == 'RND':
        x_train_k = x_train_rnd
        y_train_k = y_train_rnd
    elif data_dist == 'IID':
        x_train_k = x_train_iid
        y_train_k = y_train_iid
    else:
        x_train_k = x_train_bia
        y_train_k = y_train_bia

''''MAIN PROGRAM STARTS'''

users = [User() for i in range(no_users)]
central_modal = [tf.Variable(tf.random.truncated_normal([input_features * num_classes, 1], stddev=0.1))]

##print(central_modal[0])

# accuracies over all classes and loss
acc_train = np.zeros([n_epochs, 1])
acc_test = np.zeros([n_epochs, 1])
total_loss = np.zeros([n_epochs, 1])

# for test accuracies of specific classes
acc_test_cls0 = np.zeros([n_epochs, 1])
acc_test_cls1 = np.zeros([n_epochs, 1])
acc_test_cls2 = np.zeros([n_epochs, 1])
acc_test_cls5 = np.zeros([n_epochs, 1])
acc_test_cls7 = np.zeros([n_epochs, 1])
acc_test_cls8 = np.zeros([n_epochs, 1])

# assign the initial model to all users
for i in range(no_users):
    users[i].W1.assign(tf.reshape(central_modal[0], [input_features, num_classes]))

'''TRAINING BEGINS'''
for k in range(n_epochs):
    print('Iteration: ', k)
    batch_x = []
    batch_y = []
    batch_indexes = []

    # take the mini batches to conduct the training for this iteration
    j = 0
    for i in range(no_users):
        if i in fraud_users:
            batch_xx, batch_yy, ind = mini_batches_ind_allflip(x_train_k[i], y_train_k[i], mb_size, flipped_indexes[j])
            j = j + 1
        else:
            batch_xx, batch_yy, ind = mini_batches_ind(x_train_k[i], y_train_k[i], mb_size)
        batch_x.append(batch_xx)
        batch_y.append(batch_yy)
        batch_indexes.append(ind)

    # function to check how many poisoned samples
    if poisoned_flag == 1:
        num = len(flipped_indexes)
        flipped_samples = []
        for i in range(num):
            flipped_samples.append(set(batch_indexes[i]) & set(flipped_indexes[i][0]))
        # print(flipped_samples)

    # do SGD and find gradients and the aggregated loss for all users
    for i in range(no_users):
        gradients1, loss = get_gradients(batch_x[i], batch_y[i], users[i].W1)
        users[i].gW1.assign(gradients1)
        if i in benign_users:
            total_loss[k] = total_loss[k] + loss

    # use the gradient to update the local models
    local_updates = []
    local_updates_tr = []
    for i in range(no_users):
        local_update = central_modal[0] - lr * tf.reshape(users[i].gW1, [input_features * num_classes, 1])
        local_updates.append(local_update.numpy())
        a = tf.transpose(local_update)
        local_updates_tr.append(a[0].numpy())




    if defense_flag == 1:


        #network_averages, net_selected_counts, net_id = novel_defense_device_layer(local_updates, no_users, l2_dist_users)
        #l0_avg = novel_defense_top_layer(network_averages, net_selected_counts, central_modal)

        #central_modal[0] = l0_avg.astype(np.float32)
        # print('THIS WILL NOT HAPPEN')

        # do the testing

        next_models = []
        for i in range(no_users):

            # the method should differ from a fraud user and a benign user

            # print('CHECKING DEFENSE')
            neighbour_indexes = np.where(graph[k][i, :] == 1)[0]
            temp_user = User()
            train_acc_temp = []
            update_subset = []

            for j in neighbour_indexes:
                update_subset.append(local_updates_tr[j])
                temp_user.W1.assign(tf.reshape(local_updates_tr[j], [784, 10]))
                train_pred_temp = temp_user.neural_net(batch_x[i])
                train_acc_temp.append(accuracy(train_pred_temp, batch_y[i]))
            own_index = np.where(neighbour_indexes == i)[0][0]
            own_acc = tf.get_static_value(train_acc_temp[own_index])
            higher_acc_indexes = np.where(np.array(train_acc_temp) >= own_acc)[0].tolist()
            filtered_updates = [update_subset[i] for i in higher_acc_indexes]



            next_model = sum(filtered_updates) / len(filtered_updates)
            next_models.append(next_model)
        next_models_array = np.asarray(next_models)


            #users[i].W1.assign(tf.reshape(next_model, [784, 10]))
            # do the testing

        # compute accuracy

        # calculate score based oan accuracy

        # the score will be the weights for aggregation







    else:
        steps = 1
        update_array = np.asarray(local_updates_tr)
        #temp_arr = [5, 8, 3, 6, 10, 7]
        #con = copy.deepcopy(connections)
        #for j in range(5):
        #    np.append(connections, con, axis=0)
        #con_new = np.repeat(connections, 3, axis=1)
        if dynamic == 1:
            weights = graph[k] / graph[k].sum(axis=1)[:, None]
        else:
            weights = graph[0] / graph[0].sum(axis=1)[:, None]
        for i in range(steps):
            update_array = np.matmul(weights, update_array)

        next_models_array = update_array
        # averaging the model updates from down to up, layer 0 is the top server
        #l1_avg, l1_counts = agg_device_layer(local_updates, l2_dist_users)
        #l0_avg = agg_top_layer(l1_avg, l1_counts)

        # Update central model

        # because we expect everyone will have the same model after the training
        # central_modal[0] = np.reshape(update_array[0], (7840,1)).astype(np.float32)





    # Send updated model to the users
    for i in range(no_users):
        central_modal[0] = np.reshape(next_models_array[i], (7840, 1)).astype(np.float32)
        users[i].W1.assign(tf.reshape(central_modal[0], [784, 10]))

    '''INFERENCE'''
    # we can check with first user as every user has the same model at the inference
    # if it was training predictions then, user wise application is must

    train_acc = []
    for j in range(no_users):
        train_pred = users[j].neural_net(batch_x[j])
        train_acc.append(accuracy(train_pred, batch_y[j]))

    avgAcc_Train = np.mean(train_acc)
    acc_train[k] = avgAcc_Train

    test_acc = []
    for j in benign_users:
        test_pred = users[j].neural_net(x_test_inf)
        test_acc.append(accuracy(test_pred, y_test_inf))

    avgAcc_Test = np.mean(test_acc)
    acc_test[k] = avgAcc_Test

    if k == n_epochs - 1:
        conf_mat = tf.math.confusion_matrix(tf.argmax(y_test_inf, 1), tf.argmax(test_pred, 1), num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None)
        MA = (sum(tf.linalg.tensor_diag_part(conf_mat)) - conf_mat[1, 1]) / (sum(sum(conf_mat)) - sum(conf_mat[1, :]))
        BA = (conf_mat[1, 7]) / (sum(conf_mat[1, :]))
        TP = conf_mat[1, 1]
        FP = sum(conf_mat[:, 1]) - conf_mat[1, 1]
        TN = sum(tf.linalg.tensor_diag_part(conf_mat)) - conf_mat[1, 1]
        FN = sum(conf_mat[1, :]) - conf_mat[1, 1]

        if poisoned_flag == 1:
            if multiple == 1:
                MA = (sum(tf.linalg.tensor_diag_part(conf_mat)) - conf_mat[1, 1] - conf_mat[5, 5]) / (sum(sum(conf_mat)) - sum(conf_mat[1, :]) - sum(conf_mat[5, :]))
                BA = (conf_mat[1, 7] + conf_mat[5, 8]) / (sum(conf_mat[1, :]) + sum(conf_mat[5, :]))
                TP = conf_mat[1, 1] + conf_mat[5, 5]
                FP = sum(conf_mat[:, 1]) - conf_mat[1, 1] + sum(conf_mat[:, 5]) - conf_mat[5, 5]
                TN = sum(tf.linalg.tensor_diag_part(conf_mat)) - conf_mat[1, 1] - conf_mat[5, 5]
                FN = sum(conf_mat[1, :]) - conf_mat[1, 1] + sum(conf_mat[5, :]) - conf_mat[5, 5]

    test_pred_cls0 = users[0].neural_net(x_test_cls0)
    acc_test_cls0[k] = accuracy(test_pred_cls0, y_test_cls0)
    test_pred_cls1 = users[0].neural_net(x_test_cls1)
    acc_test_cls1[k] = accuracy(test_pred_cls1, y_test_cls1)
    test_pred_cls2 = users[0].neural_net(x_test_cls2)
    acc_test_cls2[k] = accuracy(test_pred_cls2, y_test_cls2)
    test_pred_cls5 = users[0].neural_net(x_test_cls5)
    acc_test_cls5[k] = accuracy(test_pred_cls5, y_test_cls5)
    test_pred_cls7 = users[0].neural_net(x_test_cls7)
    acc_test_cls7[k] = accuracy(test_pred_cls7, y_test_cls7)
    test_pred_cls8 = users[0].neural_net(x_test_cls8)
    acc_test_cls8[k] = accuracy(test_pred_cls8, y_test_cls8)

print(conf_mat)

Accuracy = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Recall * Precision / (Recall + Precision)

# Create accuracy plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(range(n_epochs), acc_train, 'r-*', label='Avg Tr Acc')
ax.plot(range(n_epochs), acc_test, 'b-*', label='Avg Tst Acc')

ax.plot(range(n_epochs), acc_test_cls0, 'g-*', label='Tst Acc 0')
ax.plot(range(n_epochs), acc_test_cls1, 'c-*', label='Tst Acc 1')
ax.plot(range(n_epochs), acc_test_cls2, 'm-*', label='Tst Acc 2')
ax.plot(range(n_epochs), acc_test_cls5, 'g-*', label='Tst Acc 5')
ax.plot(range(n_epochs), acc_test_cls7, 'k-*', label='Tst Acc 7')
ax.plot(range(n_epochs), acc_test_cls8, 'y-*', label='Tst Acc 8')

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.show()

if data_dist == 'BIA':
    np.savetxt('count_ones_bia_alpha(%d).txt' % alpha, count_ones, delimiter=',')
    print(count_ones)

# Create loss plot with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(range(n_epochs), total_loss / len(benign_users), 'r-*', label='Average Loss')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.yscale('log')
plt.show()

if poisoned_flag == 0:
    print(data_dist, config_text, fraud_type, ',', int(100 * poi_user_perc), ',', MA.numpy() * 100, ',', BA.numpy() * 100, ',', acc_test_cls1[-1][0] * 100, ',Server', n_epochs, 'iter seed', seed, ',', Accuracy.numpy() * 100, ',', Precision.numpy() * 100, ',', Recall.numpy() * 100, ',', F1.numpy() * 100)
else:
    if multiple == 0:
        print(data_dist, config_text, fraud_type, ',', int(100 * poi_user_perc), ',', MA.numpy() * 100, ',', BA.numpy() * 100, ',', acc_test_cls1[-1][0] * 100, ',Server', n_epochs, 'iter seed', seed, ',', Accuracy.numpy() * 100, ',', Precision.numpy() * 100, ',', Recall.numpy() * 100, ',', F1.numpy() * 100)
    else:
        print(data_dist, config_text, fraud_type, ',', int(100 * poi_user_perc), ',', MA.numpy() * 100, ',', BA.numpy() * 100, ',', acc_test_cls1[-1][0] * 100, ',', acc_test_cls5[-1][0] * 100, ',Server', n_epochs, 'iter seed', seed, ',', Accuracy.numpy() * 100, ',', Precision.numpy() * 100, ',', Recall.numpy() * 100, ',', F1.numpy() * 100)



