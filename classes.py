import tensorflow as tf

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

    '''def neural_net(self, x):
        y1 = tf.matmul(x, self.W1)
        return tf.nn.softmax(y1)'''