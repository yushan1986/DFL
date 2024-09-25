import numpy as np  # linear algebra
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class FrameworkCFL:
    def __init__(self, usrs, alp, least_smpls, in_feat, num_cls):
        self.no_users = usrs
        self.alpha = alp
        self.least_samples = least_smpls
        self.input_features = in_feat
        self.num_classes = num_cls

    def make_random_data(self, tr_data):
        # Distribute data randomly to each client. Ultimately the workers have randomly distributed data
        # Output is a list containing one numpy array for each user
        tr_data = tr_data.sample(frac=1).reset_index(drop=True)  # random shuffle the rows of train data
        X = np.array_split((tr_data.iloc[:, :-1]).to_numpy(), self.no_users)
        Y = np.array_split((tr_data.iloc[:, -1]).to_numpy(), self.no_users)
        for i in range(self.no_users):
            Y[i] = tf.keras.utils.to_categorical(Y[i], self.num_classes)
        return X, Y  # X and Y are list of numpy arrays. Y in categorical form

    def make_iid_data(self, tr_data):
        # Distribute the data from each class to its respective clients. Ultimately the workers have IID train data
        # Output is a list containing one numpy array for each user

        # Separate the training and test data into classes. Output is a list of numpy arrays for both train and test
        train_ordered = [[] for _ in range(self.num_classes)]

        for i in range(self.num_classes):
            train_ordered[i] = tr_data[tr_data[self.input_features] == i]

        X = [[] for _ in range(self.no_users)]
        Y = [[] for _ in range(self.no_users)]
        for i in range(self.no_users):
            X[i] = np.empty((0, self.input_features), dtype=float)
            Y[i] = np.empty((0, self.num_classes), dtype=float)

        for i in range(self.num_classes):
            xsplits = np.array_split((train_ordered[i].iloc[:, :-1]).to_numpy(), self.no_users)
            ysplits = np.array_split((train_ordered[i].iloc[:, -1]).to_numpy(), self.no_users)
            for j in range(self.no_users):
                X[j] = np.append(X[j], xsplits[j], axis=0)
                Y[j] = np.append(Y[j], tf.keras.utils.to_categorical(ysplits[j], self.num_classes), axis=0)
        return X, Y  # X and Y are list of numpy arrays. Y in categorical form

    def make_non_iid_data(self, tr_data):
        train_data_content = tr_data.iloc[:, :-1]
        train_data_labels = tr_data.iloc[:, -1]

        X = [[] for _ in range(self.no_users)]
        Y = [[] for _ in range(self.no_users)]
        statistic = [[] for _ in range(self.no_users)]

        dataidx_map = {}

        min_size = 0
        K = self.num_classes
        N = len(train_data_labels)

        cnt = 0
        while min_size < self.least_samples:
            cnt = cnt + 1
            print(cnt)
            idx_batch = [[] for _ in range(self.no_users)]
            for k in range(K):
                idx_k = np.where(train_data_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.no_users))
                proportions1 = np.array([p * (len(idx_j) < N / self.no_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions2 = proportions1 / proportions1.sum()
                proportions3 = (np.cumsum(proportions2) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions3))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                print(min_size)

        for j in range(self.no_users):
            dataidx_map[j] = idx_batch[j]

        count_ones = []
        for client in range(self.no_users):
            idxs = dataidx_map[client]
            X[client] = train_data_content.iloc[idxs]
            Y[client] = train_data_labels.iloc[idxs]

            if 1 in np.unique(Y[client]):
                count_ones.append(int(sum(Y[client] == 1)))
            else:
                count_ones.append(0)

            for i in np.unique(Y[client]):
                statistic[client].append((int(i), int(sum(Y[client] == i))))

        for client in range(self.no_users):
            print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(Y[client]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)

        print(count_ones)

        for i in range(self.no_users):
            X[i] = X[i].to_numpy()
            Y[i] = Y[i].to_numpy()
            Y[i] = tf.keras.utils.to_categorical(Y[i], K)

        return X, Y, count_ones  # X and y are list of numpy arrays. y in categorical form

    def create_user_data(self, dist, tr_data):
        c = []
        if dist == 'RND':
            # Random Data Preparation
            X, Y = self.make_random_data(tr_data)
        elif dist == 'IID':
            # IID Data Preparation
            X, Y = self.make_iid_data(tr_data)
        elif dist == 'BIA':
            # Non-IID Data Preparation
            X, Y, c = self.make_non_iid_data(tr_data)
        return X, Y, c


class BenUserCFL:
    def __init__(self, user_id, mal_flag, input_features, num_cls, lr, n_epochs):
        self.user_id = user_id
        self.mal_flag = mal_flag
        self.x_data = []
        self.y_data = []
        self.gradients = 0
        self.model = []
        self.num_classes = num_cls

        self.optimizer = SGD(lr=lr, decay=lr / n_epochs, momentum=0.9)
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(input_features, self.num_classes)
        local_model.compile(optimizer=SGD(lr=lr, decay=lr / n_epochs, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = local_model

    def get_data(self, x, y):
        self.x_data = x
        self.y_data = y

    def compute_gradients(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data)).batch(32)

        # Accumulate gradients across all batches
        my_grads = None

        for batch_x, batch_y in dataset:
            batch_grads = self.compute_batch_gradients(batch_x, batch_y)
            if my_grads is None:
                my_grads = batch_grads
            else:
                my_grads = [g1 + g2 for g1, g2 in zip(my_grads, batch_grads)]

        # Normalize gradients by the number of batches
        my_grads = [g / len(dataset) for g in my_grads]
        self.gradients = my_grads

    def compute_gradients_update_model(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data)).batch(32)

        # Accumulate gradients across all batches
        my_grads = None

        for batch_x, batch_y in dataset:
            batch_grads = self.compute_batch_gradients(batch_x, batch_y)
            # self.optimizer.apply_gradients(zip(batch_grads, self.model.trainable_weights))
            if my_grads is None:
                my_grads = batch_grads
            else:
                my_grads = [g1 + g2 for g1, g2 in zip(my_grads, batch_grads)]

        # Normalize gradients by the number of batches
        my_grads = [g / len(dataset) for g in my_grads]
        self.optimizer.apply_gradients(zip(my_grads, self.model.trainable_weights))
        self.gradients = my_grads

    def compute_batch_gradients(self, x, y):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        return gradients

    def train(self):
        self.model.fit(self.x_data, self.y_data, epochs=1, verbose=0)

    def send_gradients(self):
        return self.gradients

    def send_weights(self):
        return self.model.get_weights()

    def flatten_my_prams(self, mdl):
        params = [param.numpy().flatten() for param in mdl.trainable_weights]
        flattened_params = np.concatenate(params)
        return flattened_params

    def set_model_weights_from_flattened(self, flattened_weights):
        # Index to keep track of the current position in the flattened array
        index = 0
        new_weights = []

        for layer in self.model.layers:
            # Get the weights and biases of the layer (weights and biases are both in trainable_weights)
            layer_weights = layer.get_weights()
            # For each weight tensor (weights and biases) in the layer
            for weight in layer_weights:
                # Get the shape of the current weight tensor
                weight_shape = weight.shape
                # Calculate the size (number of elements) of the current weight tensor
                weight_size = np.prod(weight_shape)
                # Slice the flattened array for the current weight tensor
                new_weight_values = flattened_weights[index:index + weight_size]
                # Reshape the sliced array to the original weight tensor shape
                new_weight_values = new_weight_values.reshape(weight_shape)
                # Append to the list of new weights for this layer
                new_weights.append(new_weight_values)
                # Move the index forward by the size of the current weight tensor
                index += weight_size
            # Set the new weights for the current layer
            layer.set_weights(new_weights)
            new_weights = []  # Reset for the next layer


class MalUserCFL(BenUserCFL):
    def __init__(self, user_id, mal_flag, input_features, num_cls, lr, n_epochs):
        super().__init__(user_id, mal_flag, input_features, num_cls, lr, n_epochs)
        self.flipped_indexes = []

    def label_flip(self, source, target):
        labels = np.argmax(self.y_data, axis=1)
        indexes = np.where(labels == source)[0]
        self.flipped_indexes.extend(list(indexes))
        flipdata = tf.keras.utils.to_categorical(target, self.num_classes)
        for i in indexes:
            self.y_data[i] = flipdata

    def model_replace(self):
        poi_w = np.loadtxt('MNIST_poisoned_model17_l1_weights.txt', delimiter=',')
        poi_b = np.loadtxt('MNIST_poisoned_model17_l1_biases.txt')
        self.model.set_weights([poi_w, poi_b])

    def gauss_attack(self, m, stdv):
        noise_w = np.random.normal(m, stdv, self.model.get_weights()[0].shape)
        noise_b = np.random.normal(m, stdv, self.model.get_weights()[1].shape)
        poi_w = self.model.get_weights()[0] + noise_w
        poi_b = self.model.get_weights()[1] + noise_b
        self.model.set_weights([poi_w, poi_b])


class ServerCFL:
    def __init__(self, input_features, num_cls, lr, n_epochs):
        self.avg_gradients = 0
        self.testdata = []
        self.testacc = 0
        self.testloss = 0
        self.testpreds = []
        self.confmat = 0
        self.class_wise_acc = 0
        self.num_classes = num_cls

        self.optimizer = SGD(lr=lr, decay=lr / n_epochs, momentum=0.9)
        smlp_global = SimpleMLP()
        global_model = smlp_global.build(input_features, self.num_classes)
        global_model.compile(self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = global_model

    def get_test_data(self, t):
        self.testdata = t

    def average_apply_gradients(self, client_gradients):
        average_grads = []
        for grads in zip(*client_gradients):
            avg_grad = tf.reduce_mean(grads, axis=0)
            average_grads.append(avg_grad)
        self.avg_gradients = average_grads
        self.optimizer.apply_gradients(zip(average_grads, self.model.trainable_weights))

    def fedavg_user_models(self, user_model_weight_list):
        weights = [w[0] for w in user_model_weight_list]
        biases = [b[1] for b in user_model_weight_list]
        average_weights = np.mean(weights, axis=0)
        average_biases = np.mean(biases, axis=0)
        average = [average_weights, average_biases]
        self.model.set_weights(average)

    def test_model(self, comm_r, num_cls):
        for (x_tst, y_tst) in self.testdata:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            logits = self.model.predict(x_tst)
            loss = cce(tf.keras.utils.to_categorical(y_tst, num_cls), logits)
            acc = accuracy_score(tf.argmax(logits, axis=1), y_tst)
            print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_r, acc, loss))
            self.testpreds = logits
            self.testacc = acc
            self.testloss = loss
            self.confmat = tf.math.confusion_matrix(y_tst, tf.argmax(self.testpreds, 1), num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None)

    def calc_class_wise_acc(self, num_cls):
        acc = []
        for i in range(num_cls):
            acc.append(self.confmat[i, i] / sum(self.confmat[i, :]))
        self.class_wise_acc = acc


class SimpleMLP:

    @staticmethod
    def build(shape, classes):
        model = Sequential()
        '''
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))'''
        model.add(Dense(classes, input_shape=(shape,)))
        model.add(Activation("softmax"))
        return model


class PlotterCFL:
    def __init__(self, ephs, acc_t, t_loss, acc_1, acc_7, acc_5, acc_8):
        self.n_epochs = ephs
        self.test_acc = acc_t
        self.tot_loss = t_loss
        self.cls1_acc = acc_1
        self.cls7_acc = acc_7
        self.cls5_acc = acc_5
        self.cls8_acc = acc_8

    def plot_acc(self):
        # Create average accuracy plot over all users with pre-defined labels.
        fig, ax = plt.subplots()

        ax.plot(range(self.n_epochs), self.test_acc, 'b-*', label='Avg Tst Acc')
        ax.plot(range(self.n_epochs), (1 - self.test_acc), 'r-*', label='Max test error')

        ax.plot(range(self.n_epochs), self.cls1_acc, 'c-*', label='Tst Acc 1')
        ax.plot(range(self.n_epochs), self.cls7_acc, 'k-*', label='Tst Acc 7')
        ax.plot(range(self.n_epochs), self.cls5_acc, 'g-*', label='Tst Acc 5')
        ax.plot(range(self.n_epochs), self.cls8_acc, 'y-*', label='Tst Acc 8')

        ax.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.yticks(np.arange(0, 1.0, 0.1))
        plt.grid(True, which='both', axis='both')
        plt.show()

    def plot_loss(self):
        # Create average loss plot for all users with pre-defined labels.
        fig, ax = plt.subplots()
        ax.plot(range(self.n_epochs), self.tot_loss, 'r-*', label='Average Loss')
        ax.legend(loc='upper right', shadow=True, fontsize='x-large')
        plt.yscale('log')
        plt.grid(True, which='both', axis='both')
        plt.show()
        return 0
