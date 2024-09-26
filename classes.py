import numpy as np  # linear algebra
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

class User:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.gW1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.gW1acc = tf.Variable(tf.zeros(shape=[784, 10], dtype="float32"))
        self.gW1vec = tf.Variable(tf.zeros(shape=[1, 7840], dtype="float32"))
        self.gW1all = tf.Variable(tf.zeros(shape=[100, 7840], dtype="float32"))  # CHANGE THIS
        self.gW1var = tf.Variable(tf.zeros(shape=[1, 7840], dtype="float32"))

    def neural_net(self, x):
        y1 = tf.matmul(x, self.W1)
        return tf.nn.softmax(y1)

class User_trust:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.tempW1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.aggW1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.gW1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.tempW1vec = tf.Variable(tf.zeros(shape=[1, 7840], dtype="float32"))
        self.aggW1vec = tf.Variable(tf.zeros(shape=[1, 7840], dtype="float32"))
        self.neighbours = None
        self.neighbourUpdates = None

    def neural_net(self, x):
        y1 = tf.matmul(x, self.W1)
        return tf.nn.softmax(y1)

    def sim_dis_calc(self, nei_updates):
        cs_list = []
        euc_list = []
        for i in range(len(nei_updates)):
            cs = cosine_similarity(self.tempW1vec.numpy(), [nei_updates[i]])
            cs_list.append(cs[0][0])
            euc = pairwise_distances(self.tempW1vec.numpy(), [nei_updates[i]])
            euc_list.append(euc[0][0])
        return cs_list, euc_list

    def create_agg(self, nei_updates, weights):
        nei_sum = np.zeros(7840)
        for i in range(len(nei_updates)):
            prod = weights[i] * nei_updates[i]
            nei_sum = np.add(nei_sum, prod)
        nei_sum = tf.cast(tf.reshape(nei_sum, [784, 10]), tf.float32)
        aggmodel= (self.tempW1 + nei_sum)/2
        return aggmodel

    def calc_trust(self, cs, euc, tes):
        y = 0.76759053 * cs + 0.71847479 * euc + 0.28215525 * tes
        return y

    @staticmethod
    def accuracy(y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    @staticmethod
    def nei_tests(nei_updates, x, y):
        test_scores = []
        for i in range(len(nei_updates)):
            nei_model = tf.reshape(nei_updates[i], [784, 10])
            y1 = tf.matmul(x, nei_model)
            y2 = tf.nn.softmax(y1)
            #correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y, 1))
            test_acc = User_trust.accuracy(y2, y)
            test_scores.append(tf.get_static_value(test_acc))
        return test_scores

class User_All_Label:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([89, 30], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([30, 9], stddev=0.1))

        self.gW1 = tf.Variable(tf.random.truncated_normal([89, 30], stddev=0.1))
        self.gW2 = tf.Variable(tf.random.truncated_normal([30, 9], stddev=0.1))

        self.gWconc = tf.Variable(tf.zeros(shape=[1, 2940], dtype="float32"))

    def neural_net(self, x):
        y1 = tf.nn.relu(tf.matmul(x, self.W1))
        y2 = tf.matmul(y1, self.W2)
        return tf.nn.softmax(y2)


class User_Two_Label:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([120, 10], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([10, 2], stddev=0.1))

        self.gW1 = tf.Variable(tf.random.truncated_normal([120, 10], stddev=0.1))
        self.gW2 = tf.Variable(tf.random.truncated_normal([10, 2], stddev=0.1))

    def neural_net(self, x):
        y1 = tf.nn.relu(tf.matmul(x, self.W1))
        y2 = tf.matmul(y1, self.W2)
        return tf.nn.softmax(y2)


class User_AL_FG:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([120, 10], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([10, 23], stddev=0.1))

        self.gW1 = tf.Variable(tf.random.truncated_normal([120, 10], stddev=0.1))
        self.gW1acc = tf.Variable(tf.zeros(shape=[120, 10], dtype="float32"))
        self.gW1vec = tf.Variable(tf.zeros(shape=[1, 1200], dtype="float32"))

        self.gW2 = tf.Variable(tf.random.truncated_normal([10, 23], stddev=0.1))
        self.gW2acc = tf.Variable(tf.zeros(shape=[10, 23], dtype="float32"))
        self.gW2vec = tf.Variable(tf.zeros(shape=[1, 230], dtype="float32"))

    def neural_net(self, x):
        y1 = tf.nn.relu(tf.matmul(x, self.W1))
        y2 = tf.matmul(y1, self.W2)
        return tf.nn.softmax(y2)


class Aggregator:
    def __init__(self):
        self.W1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.gW1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
        self.gW1acc = tf.Variable(tf.zeros(shape=[784, 10], dtype="float32"))
        self.gW1vec = tf.Variable(tf.zeros(shape=[1, 7840], dtype="float32"))
        self.gW1all = tf.Variable(tf.zeros(shape=[100, 7840], dtype="float32"))  # CHANGE THIS
        self.gW1var = tf.Variable(tf.zeros(shape=[1, 7840], dtype="float32"))

class SimpleMLP_MN:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        '''
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))'''

        model.add(Dense(10, input_shape=(shape,)))

        model.add(Activation("softmax"))
        return model

class SimpleMLP_CF:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Input(shape=(shape[0], shape[1], shape[2])))
        #model.add(Lambda(lambda x: expand_dims(x, axis=-1)))
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Activation("relu"))
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Activation("relu"))
        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

    '''def neural_net(self, x):
        y1 = tf.matmul(x, self.W1)
        return tf.nn.softmax(y1)'''