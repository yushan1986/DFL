import numpy as np
import tensorflow as tf
import math
import pandas as pd
import copy
import os

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import random

num_classes = 10

def get_files_with_prefix(prefix):
    # Get the list of all files in the directory
    all_files = os.listdir('.')

    # Filter files based on the specified prefix
    filtered_files = [file for file in all_files if file.startswith(prefix)]

    return filtered_files

def upper_tri_values(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def counts_from_cum(inlist):
    outlist = []
    item = 0
    for i in inlist:
        freq = i - item
        outlist.append(freq)
        item = item + freq
    return outlist

def mini_batches(X, Y, mb_size):

    m = X.shape[0]
    perm = list(np.random.permutation(m))
    X_temp = X[perm,:]
    Y_temp = Y[perm,:].reshape((m, Y.shape[1]))

    X_r = tf.convert_to_tensor(X_temp[0:mb_size,:], dtype=np.float32)
    Y_r = tf.convert_to_tensor(Y_temp[0:mb_size,:], dtype=np.float32)
    return X_r,Y_r

def mini_batches_ind(X, Y, mb_size):

    m = X.shape[0]
    perm = list(np.random.permutation(m))
    selected_batch = perm[0:mb_size]
    X_temp = X[perm,:]
    Y_temp = Y[perm,:].reshape((m, Y.shape[1]))

    X_r = tf.convert_to_tensor(X_temp[0:mb_size,:], dtype=np.float32)
    Y_r = tf.convert_to_tensor(Y_temp[0:mb_size,:], dtype=np.float32)
    return X_r,Y_r, selected_batch

def mini_batches_ind_allflip(X, Y, mb_size, flipped_ind):
    indices_all = list(range(X.shape[0]))
    ind_not_flip = list(set(indices_all).difference(flipped_ind[0]))

    # use     `arr[tuple(seq)]`     instead     of     `arr[seq]`

    flipped_X = X[flipped_ind[0]]
    flipped_Y = Y[flipped_ind[0]]
    rem_X = X[ind_not_flip]
    rem_Y = Y[ind_not_flip]

    not_flip_samples_req = max(0, (mb_size - len(flipped_ind[0])))

    if not_flip_samples_req > 0:
        m = rem_X.shape[0]
        perm = list(np.random.permutation(m))
        X_shuf = rem_X[perm, :]
        Y_shuf = rem_Y[perm, :].reshape((m, rem_Y.shape[1]))

        selected_batch_ind = perm[0:not_flip_samples_req]
        selected_batch_ind.extend(list(flipped_ind[0]))  # now this has all selected indexes

        benign_X = X_shuf[0:not_flip_samples_req, :]
        benign_Y = Y_shuf[0:not_flip_samples_req, :]

        X_temp = np.concatenate((flipped_X, benign_X), axis=0)
        Y_temp = np.concatenate((flipped_Y, benign_Y), axis=0)
    else:
        selected_batch_ind = random.choices(flipped_ind[0], k=mb_size)
        X_temp = X[selected_batch_ind]
        Y_temp = Y[selected_batch_ind]

    X_r = tf.convert_to_tensor(X_temp, dtype=np.float32)
    Y_r = tf.convert_to_tensor(Y_temp, dtype=np.float32)

    return X_r, Y_r, selected_batch_ind

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return -tf.reduce_sum(y_true * tf.math.log(y_pred))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def get_gradients(x, y, W1):
    # Variables to update, i.e. trainable variables.
    trainable_variables = W1  # removed square brackets

    with tf.GradientTape() as g:
        g.watch(W1)  # removed square brackets
        y1 = tf.matmul(x, W1)
        pred = tf.nn.softmax(y1)
        loss = cross_entropy(pred, y)

        # Compute gradients.
    gradients1 = g.gradient(loss, trainable_variables)

    return gradients1, loss

    # Compute gradients.
    gradients1, gradients2 = g.gradient(loss, trainable_variables)
    return gradients1, gradients2, loss

# Flips y labels
# indata is a copy of y_train_sep
# unserlist is the list of users to flip lables
# sourcelist is the source labels to flip. carefully assign this in case of biased data
# sourcelist is the list of destination labels
# perventage is how much of source labels to be flipped
def label_flip_multiple(indata, userlist, sourcelist, destlist, percentage):
    perc = percentage
    fip_item = 0
    for i in userlist:
        inds = np.argmax(indata[i], axis=1)
        indexes = np.where(inds == sourcelist[fip_item])
        flipdata = tf.keras.utils.to_categorical(destlist[fip_item], num_classes)
        fip_item = fip_item + 1
        for j in indexes[0]:
            indata[i][j] = flipdata
    return indata

def label_flip_multiple_ind(indata, userlist, sourcelist, destlist, percentage):
    perc = percentage
    fip_item = 0
    flipped_indexes = []
    for i in userlist:
        inds = np.argmax(indata[i], axis=1)
        indexes = np.where(inds == sourcelist[fip_item])
        flipped_indexes.append(list(indexes))
        flipdata = tf.keras.utils.to_categorical(destlist[fip_item], num_classes)
        fip_item = fip_item + 1
        for j in indexes[0]:
            indata[i][j] = flipdata
    return indata, flipped_indexes

def angle(u, v):
    cosine = dot(u, v) / (norm(u) * norm(v))  # -> cosine of the angle
    radians = arccos(clip(cosine, -1, 1)) # if you really want the angle
    #print(radians)
    return radians * 180 / (math.pi)

def pardon(cs_sim, no_users):

    # calculate v vlaues
    np.fill_diagonal(cs_sim, -1)
    v_values = cs_sim.max(1)

    # pardoning
    cs_sim_mod = copy.deepcopy(cs_sim)
    for i in range(no_users):
        for j in range(i+1,no_users):
            if v_values[j] > v_values[i]:
                cs_sim_mod[i,j] = cs_sim[i,j]*v_values[i]/(v_values[j])
                cs_sim_mod[j,i] = cs_sim[i,j]*v_values[i]/(v_values[j])
    np.fill_diagonal(cs_sim_mod, -1)
    cs_values = cs_sim_mod.max(1)
    alpha = np.ones(no_users) - cs_values
    alpha_modified = alpha/np.max(alpha)
    return alpha_modified

def make_biased_data(train_data, no_users, alpha, least_samples):

    train_data_content = train_data.iloc[:, :-1]
    train_data_labels = train_data.iloc[:, -1]

    X = [[] for _ in range(no_users)]
    Y = [[] for _ in range(no_users)]
    statistic = [[] for _ in range(no_users)]

    dataidx_map = {}

    min_size = 0
    K = num_classes
    N = len(train_data_labels)

    cnt = 0
    while min_size < least_samples:
        cnt = cnt + 1
        print(cnt)
        idx_batch = [[] for _ in range(no_users)]
        for k in range(K):
            idx_k = np.where(train_data_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, no_users))
            proportions1 = np.array([p * (len(idx_j) < N / no_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions2 = proportions1 / proportions1.sum()
            proportions3 = (np.cumsum(proportions2) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions3))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            print(min_size)

    for j in range(no_users):
        dataidx_map[j] = idx_batch[j]

    count_ones = []
    for client in range(no_users):
        idxs = dataidx_map[client]
        X[client] = train_data_content.iloc[idxs]
        Y[client] = train_data_labels.iloc[idxs]

        if 1 in np.unique(Y[client]):
            count_ones.append(int(sum(Y[client] == 1)))
        else:
            count_ones.append(0)

        for i in np.unique(Y[client]):
            statistic[client].append((int(i), int(sum(Y[client] == i))))


    for client in range(no_users):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(Y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    print(count_ones)

    for i in range(no_users):
        X[i] = X[i].to_numpy()
        Y[i] = Y[i].to_numpy()
        Y[i] = tf.keras.utils.to_categorical(Y[i], num_classes)

    return X, Y, count_ones # X and y are list of numpy arrays. y in categorical form

def make_random_data(train_data, no_users, num_cls):
    # Distribute data randomly to each client. Ultimately the workers have randomly distributed data
    # Output is a list containing one numpy array for each user
    train_data = train_data.sample(frac=1).reset_index(drop=True)   # random shuffle the rows of train data
    X = np.array_split((train_data.iloc[:, :-1]).to_numpy(), no_users)
    Y = np.array_split((train_data.iloc[:, -1]).to_numpy(), no_users)
    for i in range(no_users):
        Y[i] = tf.keras.utils.to_categorical(Y[i], num_cls)
    return X, Y  # X and Y are list of numpy arrays. Y in categorical form

def make_iid_data(train_ordered, no_users, input_features, num_cls):
    # Distribute the data from each class to its respective clients. Ultimately the workers have IID train data
    # Output is a list containing one numpy array for each user
    X = [[] for i in range(no_users)]
    Y = [[] for i in range(no_users)]
    for i in range(no_users):
        X[i] = np.empty((0, input_features), dtype=float)
        Y[i] = np.empty((0, num_cls), dtype=float)

    for i in range(num_cls):
        xsplits = np.array_split((train_ordered[i].iloc[:, :-1]).to_numpy(), no_users)
        ysplits = np.array_split((train_ordered[i].iloc[:, -1]).to_numpy(), no_users)
        for j in range(no_users):
            X[j] = np.append(X[j], xsplits[j], axis=0)
            Y[j] = np.append(Y[j], tf.keras.utils.to_categorical(ysplits[j], num_cls), axis=0)
    return X, Y  # X and Y are list of numpy arrays. Y in categorical form


'''AGGREGATION FUNCTIONS'''

# aggregates the lower layer updates based on the distribution of the nodes
# and return the averages at the higher layer
def agg_device_layer(lower_layer_updates, distribution):
    start_ind = 0
    upper_layer_avgs = []
    dev_layer_count = []
    for i in range(len(distribution)):
        edge_agg = 0
        k = 0
        for j in range(start_ind, start_ind + distribution[i]):
            # here we have to apply the robust aggregation rule
            edge_agg = edge_agg + lower_layer_updates[j]
            k = k+1
        start_ind = start_ind + distribution[i]
        avg = edge_agg / distribution[i]
        upper_layer_avgs.append(avg)
        dev_layer_count.append(k)
    return upper_layer_avgs, dev_layer_count

def agg_int_layer(lower_layer_updates, distribution, lower_layer_weights):
    start_ind = 0
    upper_layer_avgs = []
    layer_count = []
    for i in range(len(distribution)):
        edge_agg = 0
        tot_devices = 0
        for j in range(start_ind, start_ind + distribution[i]):
            edge_agg = edge_agg + lower_layer_updates[j] * lower_layer_weights[j]
            tot_devices = tot_devices + lower_layer_weights[j]
        start_ind = start_ind + distribution[i]
        if tot_devices == 0:
            avg = 0
        else:
            avg = edge_agg / tot_devices
        upper_layer_avgs.append(avg)
        layer_count.append(tot_devices)
    return upper_layer_avgs, layer_count

def agg_top_layer(lower_layer_updates, lower_layer_weights):
    edge_agg = 0
    tot_devices = 0
    for i in range(len(lower_layer_updates)):
        edge_agg = edge_agg + lower_layer_updates[i] * lower_layer_weights[i]
        tot_devices = tot_devices + lower_layer_weights[i]
    if tot_devices == 0:
        avg = 0
    else:
        avg = edge_agg / tot_devices
    return avg

def agg_int_layer1(lower_layer_updates, distribution):
    start_ind = 0
    upper_layer_avgs = []
    for i in range(len(distribution)):
        edge_agg = 0
        tot_devices = 0
        for j in range(start_ind, start_ind + distribution[i]):
            # here we have to apply the robust aggregation rule
            edge_agg = edge_agg + lower_layer_updates[j]
        start_ind = start_ind + distribution[i]
        avg = edge_agg / distribution[i]
        upper_layer_avgs.append(avg)
    return upper_layer_avgs

def agg_top_layer1(lower_layer_updates):
    edge_agg = 0
    for i in range(len(lower_layer_updates)):
        edge_agg = edge_agg + lower_layer_updates[i]
    avg = edge_agg / len(lower_layer_updates)
    return avg

def agg_device_layer_krum(lower_layer_updates, device_distribution, fraudster_distribution):
    update_len = lower_layer_updates[0].shape[0]    # get the length of the weights
    start_ind = 0
    upper_layer_model = []
    dev_layer_count = []
    for i in range(len(device_distribution)):
        max_devices_to_check = device_distribution[i] - fraudster_distribution[i] - 2
        if max_devices_to_check <= 0:   # handle the case where all devices are fraudsters
            upper_layer_model.append(0)
            dev_layer_count.append(0)
            start_ind = start_ind + device_distribution[i]
        else:   # if there is at least one legitimate device
            update_array = np.empty((update_len, 0), dtype=np.float32)

            for j in range(start_ind, start_ind + device_distribution[i]):
                update_array = np.append(update_array, lower_layer_updates[j], axis=1)  # collect the updates into an array
            dis_mat = pairwise_distances(np.transpose(update_array)) # calculate pairwise distance matrix

            # calculate the sums of ordered pairwise distances
            sums = []
            for k in range(dis_mat.shape[0]):
                arr1 = np.sort(dis_mat[k, :])
                sum1 = sum(arr1[:max_devices_to_check + 1])
                sums.append(sum1)

            # select the model with minimum sum and consider it to send to the above layer
            min_index = sums.index(min(sums))
            upper_layer_model.append(lower_layer_updates[start_ind + min_index])
            dev_layer_count.append(1)

            start_ind = start_ind + device_distribution[i]
    return upper_layer_model, dev_layer_count

def agg_median(lower_layer_updates, device_distribution):
    update_len = lower_layer_updates[0].shape[0]    # get the length of the weights
    update_wid = lower_layer_updates[0].shape[1]  # get the width of the weights
    start_ind = 0
    upper_layer_model = []
    dev_layer_count = []
    for i in range(len(device_distribution)):
        k = 0
        update_array = np.empty((update_len, update_wid, device_distribution[i]), dtype=np.float32)
        for j in range(start_ind, start_ind + device_distribution[i]):
            update_array[:,:,k] = lower_layer_updates[j]  # collect the updates into an array
            k = k + 1
        median = np.median(update_array, axis=2)
        start_ind = start_ind + device_distribution[i]
        upper_layer_model.append(median)
        dev_layer_count.append(1)
    return upper_layer_model, dev_layer_count

def agg_device_layer_foolsgold_w(lower_layer_updates, device_distribution):
    const = 1
    start_ind = 0
    upper_layer_avgs = []
    dev_layer_count = []
    for i in range(len(device_distribution)):
        lower_layer_updates_mod = []
        # prepare updates to calculate similarity
        for j in range(start_ind, start_ind + device_distribution[i]):
            lower_layer_updates_mod.append(np.reshape(lower_layer_updates[j], (1, 7840))[0])
        # calculate cosine similarity
        cs_sim = cosine_similarity(lower_layer_updates_mod)
        # pardoning
        alpha_modified = pardon(cs_sim, device_distribution[i])
        # Calculate the per-user learning rate for the next iteration
        final = np.zeros(device_distribution[i])
        for k in range(device_distribution[i]):
            # address the two special cases 0 and 1
            if alpha_modified[k] == 1 or alpha_modified[k] == 0:
                final[k] = alpha_modified[k]
            else:
                final[k] = const*(np.log(alpha_modified[k]/(1-alpha_modified[k]))+0.5)
        learning_rates = np.clip(final, 0, 1)
        valid_users = device_distribution[i] - learning_rates[np.where(learning_rates == 0)].size
        # Calculate the model update
        update = 0
        for l in range(device_distribution[i]):
            update = update + learning_rates[l] * lower_layer_updates[l + start_ind]
        start_ind = start_ind + device_distribution[i]
        if valid_users == 0:
            avg = 0
        else:
            avg = update / valid_users
        upper_layer_avgs.append(avg)
        dev_layer_count.append(valid_users)
    return upper_layer_avgs, dev_layer_count

def agg_int_layer_foolsgold_w(lower_layer_updates, distribution, lower_layer_weights):
    const = 1
    start_ind = 0
    upper_layer_avgs = []
    layer_count = []
    for i in range(len(distribution)):
        lower_layer_updates_mod = []
        tot_devices = 0
        for j in range(start_ind, start_ind + distribution[i]):
            lower_layer_updates_mod.append(np.reshape(lower_layer_updates[j], (1, 7840))[0])
        # calculate cosine similarity
        cs_sim = cosine_similarity(lower_layer_updates_mod)
        # pardoning
        alpha_modified = pardon(cs_sim, distribution[i])
        # Calculate the per-user learning rate for the next iteration
        final = np.zeros(distribution[i])
        for k in range(distribution[i]):
            # address the two special cases 0 and 1
            if alpha_modified[k] == 1 or alpha_modified[k] == 0:
                final[k] = alpha_modified[k]
            else:
                final[k] = const*(np.log(alpha_modified[k]/(1-alpha_modified[k]))+0.5)
        learning_rates = np.clip(final, 0, 1)
        # valid_users = distribution[i] - learning_rates[np.where(learning_rates == 0)].size
        # Calculate the model update
        update = 0
        for l in range(distribution[i]):
            update = update + learning_rates[l] * lower_layer_updates[l + start_ind] * lower_layer_weights[l + start_ind]
            tot_devices = tot_devices + learning_rates[l] * lower_layer_weights[l + start_ind]
        start_ind = start_ind + distribution[i]
        if tot_devices == 0:
            avg = 0
        else:
            avg = update / tot_devices
        upper_layer_avgs.append(avg)
        layer_count.append(tot_devices)
    return upper_layer_avgs, layer_count

def agg_top_layer_foolsgold_w(lower_layer_updates, lower_layer_weights):
    const = 1
    no_users = len(lower_layer_updates)
    lower_layer_updates_mod = []
    tot_devices = 0
    for i in range(no_users):
        lower_layer_updates_mod.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])
    # calculate cosine similarity
    cs_sim = cosine_similarity(lower_layer_updates_mod)
    # pardoning
    alpha_modified = pardon(cs_sim, no_users)
    final = np.zeros(no_users)
    for k in range(no_users):
        # address the two special cases 0 and 1
        if alpha_modified[k] == 1 or alpha_modified[k] == 0:
            final[k] = alpha_modified[k]
        else:
            final[k] = const * (np.log(alpha_modified[k] / (1 - alpha_modified[k])) + 0.5)
    learning_rates = np.clip(final, 0, 1)
    # valid_users = no_users - learning_rates[np.where(learning_rates == 0)].size
    # Calculate the model update
    update = 0
    for l in range(no_users):
        update = update + learning_rates[l] * lower_layer_updates[l] * lower_layer_weights[l]
        tot_devices = tot_devices + learning_rates[l] * lower_layer_weights[l]
    if tot_devices == 0:
        avg = 0
    else:
        avg = update / tot_devices
    return avg

def agg_server_foolsgold_w(lower_layer_updates):
    const = 1
    no_users = len(lower_layer_updates)
    lower_layer_updates_mod = []
    for i in range(no_users):
        lower_layer_updates_mod.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])
    # calculate cosine similarity
    cs_sim = cosine_similarity(lower_layer_updates_mod)
    # pardoning
    alpha_modified = pardon(cs_sim, no_users)
    final = np.zeros(no_users)
    for k in range(no_users):
        # address the two special cases 0 and 1
        if alpha_modified[k] == 1 or alpha_modified[k] == 0:
            final[k] = alpha_modified[k]
        else:
            final[k] = const * (np.log(alpha_modified[k] / (1 - alpha_modified[k])) + 0.5)
    learning_rates = np.clip(final, 0, 1)
    # valid_users = no_users - learning_rates[np.where(learning_rates == 0)].size
    # Calculate the model update
    update = 0
    for l in range(no_users):
        update = update + learning_rates[l] * lower_layer_updates[l]
    avg = update / no_users
    return avg

def foolsgold_gradients(lower_layer_updates, device_distribution):
    update_len = lower_layer_updates[0].shape[0]    # get the length of the weights
    update_wid = lower_layer_updates[0].shape[1]  # get the width of the weights
    start_ind = 0
    upper_layer_model = []
    dev_layer_count = []
    for i in range(len(device_distribution)):
        k = 0
        update_array = np.empty((update_len, update_wid, device_distribution[i]), dtype=np.float32)
        for j in range(start_ind, start_ind + device_distribution[i]):
            update_array[:,:,k] = lower_layer_updates[j]  # collect the updates into an array
            k = k + 1
        median = np.median(update_array, axis=2)
        start_ind = start_ind + device_distribution[i]
        upper_layer_model.append(median)
        dev_layer_count.append(1)
    return upper_layer_model, dev_layer_count

def group_agg_device_layer(lower_layer_updates, distribution):
    local_updates_for_cs = []
    no_users = sum(distribution)
    networks = len(distribution)
    for i in range(no_users):
        local_updates_for_cs.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])

    start_ind = 0
    group_averages = []
    group_device_count = []
    orig_network_id = []
    cs_similarity = []
    for i in range(networks):
        update_array = np.array(local_updates_for_cs[start_ind:start_ind + distribution[i]])
        update_array1 = lower_layer_updates[start_ind:start_ind + distribution[i]]
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(update_array)
        print(clusterer.labels_)
        max_clust_label = max(clusterer.labels_)
        if max_clust_label == -1:
            group_device_count.append(len(update_array))
            group_averages.append(np.mean(update_array1, axis=0))
            orig_network_id.append(i)

            # calculate the average of cosine similarities
        else:
            print('several clusters in the local network')
            no_of_clusters = max(clusterer.labels_) + 1
            for j in range(no_of_clusters):
                indices = np.where(clusterer.labels_ == j)[0]
                group_device_count.append(len(indices))
                group = [update_array1[i] for i in indices]
                group_avg = np.mean(group, axis=0)
                group_averages.append(group_avg)
                orig_network_id.append(i)
                # calculate the average of cosine similarities per group
        start_ind = start_ind + distribution[i]

    return 1

def newdefense_low_layers(lower_layer_updates, distribution, central_model0):

    local_updates_for_cs = []
    lower_layer_nodes = len(lower_layer_updates)
    for i in range(lower_layer_nodes):
        local_updates_for_cs.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])

    start_ind = 0
    group_averages = []
    group_device_count = []

    for i in range(len(distribution)):
        update_array_for_cluster = np.array(local_updates_for_cs[start_ind:start_ind + distribution[i]])
        update_array_for_mean = lower_layer_updates[start_ind:start_ind + distribution[i]]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        clusterer.fit(update_array_for_cluster)
        #print(clusterer.labels_)
        if max(clusterer.labels_) == -1:
            no_of_clusters = 1
        else:
            no_of_clusters = max(clusterer.labels_) + 1
        if no_of_clusters == 1:
            group_device_count.append(len(update_array_for_cluster))
            group_averages.append(np.mean(update_array_for_mean, axis=0))
        else:
            within_group_averages = []
            within_group_device_counts = []
            group_euc_dists_means = []
            for j in range(no_of_clusters):
                indices = np.where(clusterer.labels_ == j)[0]
                within_group_device_counts.append(len(indices))
                group = list(map(update_array_for_mean.__getitem__, indices))
                group_avg = np.mean(group, axis=0)
                within_group_averages.append(group_avg)
                euc_dists = []
                for l in range(len(group)):
                    eu_dist = np.linalg.norm(group[l] - central_model0)
                    euc_dists.append(eu_dist)
                group_euc_dists_means.append(np.mean(euc_dists))
            min_index_group_euc_dists = group_euc_dists_means.index(min(group_euc_dists_means))
            group_device_count.append(within_group_device_counts[min_index_group_euc_dists])
            group_averages.append(within_group_averages[min_index_group_euc_dists])
            #print(group_euc_dists_means)
        start_ind = start_ind + distribution[i]
    return group_averages

def newdefense_top_layer(lower_layer_updates, central_model0):

    local_updates_for_cs = []
    lower_layer_nodes = len(lower_layer_updates)
    for i in range(lower_layer_nodes):
        local_updates_for_cs.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])

    update_array_for_cluster = np.array(local_updates_for_cs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    clusterer.fit(update_array_for_cluster)
    #print(clusterer.labels_)
    if max(clusterer.labels_) == -1:
        no_of_clusters = 1
    else:
        no_of_clusters = max(clusterer.labels_) + 1
    if no_of_clusters == 1:
        #l0_avg1 = agg_top_layer(group_averages2, group_device_count2)
        l0_avg1 = np.mean(lower_layer_updates, axis=0)
    else:
        within_group_averages = []
        group_euc_dists_mean = []
        for j in range(no_of_clusters):
            indices = np.where(clusterer.labels_ == j)[0]
            group = list(map(lower_layer_updates.__getitem__, indices))
            group_avg = np.mean(group, axis=0)
            within_group_averages.append(group_avg)
            euc_dists = []
            for l in range(len(group)):
                eu_dist = np.linalg.norm(group[l] - central_model0)
                euc_dists.append(eu_dist)
            group_euc_dists_mean.append(np.mean(euc_dists))
        min_index_group_euc_dists = group_euc_dists_mean.index(min(group_euc_dists_mean))
        l0_avg1 = within_group_averages[min_index_group_euc_dists]
    return l0_avg1

def novel_defense_device_layer(local_updates, no_users, l2_dist_users):

    local_updates_for_cs = []
    for i in range(no_users):
        local_updates_for_cs.append(np.reshape(local_updates[i], (1, 7840))[0])

    start_ind = 0
    start_fraud_ind = 0
    start_fraud_user_ind = 0
    network_averages = []
    net_selected_counts = []
    euc_dis_means_of_selected_clusters = []
    net_id = []

    for i in range(len(l2_dist_users)):
        net_updates_to_cl = np.array(local_updates_for_cs[start_ind:start_ind + l2_dist_users[i]])
        net_updates_to_avg = local_updates[start_ind:start_ind + l2_dist_users[i]]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='leaf')
        clusterer.fit(net_updates_to_cl.astype('float64'))
        # print(clusterer.labels_)
        no_of_clusters = max(clusterer.labels_) + 1

        if no_of_clusters == 0:
            net_selected_counts.append(len(net_updates_to_cl))
            net_id.append(i)
            network_averages.append(np.mean(net_updates_to_avg, axis=0)) # only when the average is sent
        else:
            net_id.extend([i] * no_of_clusters)
            cluster_wise_averages = []
            cluster_wise_device_counts = []
            for j in range(no_of_clusters):
                indices = np.where(clusterer.labels_ == j)[0]
                cluster_wise_device_counts.append(len(indices)) # need to change this to capture the count for the selected cluster
                cluster = list(map(net_updates_to_avg.__getitem__, indices))
                #clusters_seperated.append(cluster)
                this_cluster_avg = np.mean(cluster, axis=0)
                cluster_wise_averages.append(this_cluster_avg)
            network_averages.extend(cluster_wise_averages)
            net_selected_counts.extend(cluster_wise_device_counts)
        start_ind = start_ind + l2_dist_users[i]

    return network_averages, net_selected_counts, net_id

def novel_defense_top_layer(lower_layer_updates, counts, central_modal):

    local_updates_for_cs = []
    for i in range(len(lower_layer_updates)):
        local_updates_for_cs.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])
    update_array = np.array(local_updates_for_cs)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='leaf')
    clusterer.fit(update_array)
    #print(clusterer.labels_)
    no_of_clusters = max(clusterer.labels_) + 1

    if no_of_clusters == 0:
        l0_avg = agg_top_layer(lower_layer_updates, counts)
    else:
        l1_averages = []
        cluster_wise_euc_dists_means = []
        for j in range(no_of_clusters):
            indices = np.where(clusterer.labels_ == j)[0]
            cluster = list(map(lower_layer_updates.__getitem__, indices))
            cluster_weights = list(map(counts.__getitem__, indices))
            this_cluster_avg = np.average(cluster, axis=0, weights=cluster_weights)
            l1_averages.append(this_cluster_avg)
            cluster_euc_dists = []
            for l in range(len(cluster)):
                eu_dist = np.linalg.norm(cluster[l] - central_modal[0])
                cluster_euc_dists.append(eu_dist)
            cluster_wise_euc_dists_means.append(np.mean(cluster_euc_dists))
        min_index_group_euc_dists = cluster_wise_euc_dists_means.index(min(cluster_wise_euc_dists_means))
        l0_avg = l1_averages[min_index_group_euc_dists]

    return l0_avg

def novel_defense_int_layer(lower_layer_updates, net_selected_counts, int_dist_mod):

    local_updates_for_cs2 = []
    for i in range(len(lower_layer_updates)):
        local_updates_for_cs2.append(np.reshape(lower_layer_updates[i], (1, 7840))[0])

    start_ind = 0
    l2_subnet_averages = []
    l2_selected_counts = []

    for i in range(len(int_dist_mod)):
        subnet_for_cl = np.array(local_updates_for_cs2[start_ind:start_ind + int_dist_mod[i]])
        subnet_for_avg = lower_layer_updates[start_ind:start_ind + int_dist_mod[i]]
        subnet_for_counts = net_selected_counts[start_ind:start_ind + int_dist_mod[i]]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='leaf')
        clusterer.fit(subnet_for_cl)
        ##print(clusterer.labels_)
        no_of_clusters = max(clusterer.labels_) + 1

        if no_of_clusters == 0:
            l2_selected_counts.append(sum(subnet_for_counts))
            this_cluster_avg = np.average(subnet_for_avg, axis=0, weights=subnet_for_counts)
            l2_subnet_averages.append(this_cluster_avg)
            ##print('One cluster in the subnet', i)
        else:
            ##print('several clusters in the subnet', i)
            cluster_wise_averages = []
            cluster_wise_device_counts = []
            for j in range(no_of_clusters):
                indices = np.where(clusterer.labels_ == j)[0]
                cluster = list(map(subnet_for_avg.__getitem__, indices))
                cluster_weights = list(map(subnet_for_counts.__getitem__, indices))
                cluster_wise_device_counts.append(sum(cluster_weights))
                this_cluster_avg = np.average(cluster, axis=0, weights=cluster_weights)
                cluster_wise_averages.append(this_cluster_avg)
            l2_selected_counts.extend(cluster_wise_device_counts)
            l2_subnet_averages.extend(cluster_wise_averages)
        start_ind = start_ind + int_dist_mod[i]

    return l2_subnet_averages, l2_selected_counts




