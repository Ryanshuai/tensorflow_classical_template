import tensorflow as tf
#没有BN层试验
class NET:
    def __init__(self):
        #self.collecion = ['mnist', tf.GraphKeys.GLOBAL_VARIABLES]
        xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
        xavier_init = tf.contrib.layers.xavier_initializer()

        self.keep_prob = tf.placeholder(tf.float32)
        self.target = tf.placeholder(tf.float32, [None, 10])
        self.flatten_im = tf.placeholder(tf.float32, [None, 784])
        self.input = tf.reshape(self.flatten_im, [-1, 28, 28, 1])

        # conv1
        self.W_conv1 = tf.Variable(xavier_init_conv2d([5,5,1,32]))
        self.b_conv1 = tf.constant(0., shape=[32])
        self.z_conv1 = tf.nn.conv2d(self.input / 255, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv1
        self.a_conv1 = tf.nn.relu(self.z_conv1)
        self.p_conv1 = tf.nn.max_pool(self.a_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.sum_his_W_conv1 = tf.summary.histogram('W_conv1', self.W_conv1)
        self.sum_his_b_conv1 = tf.summary.histogram('b_conv1', self.b_conv1)

        # conv2
        self.W_conv2 = tf.Variable(xavier_init_conv2d([5,5,32,64]))
        self.b_conv2 = tf.constant(0.,shape=[64])
        self.z_conv2 = tf.nn.conv2d(self.p_conv1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv2
        self.a_conv2 = tf.nn.relu(self.z_conv2)
        self.p_conv2 = tf.nn.max_pool(self.a_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.sum_his_W_conv2 = tf.summary.histogram('W_conv2', self.W_conv2)
        self.sum_his_b_conv2 = tf.summary.histogram('b_conv2', self.b_conv2)

        self.p_conv2_flatten = tf.reshape(self.p_conv2, [-1, 7 * 7 * 64])

        # fc1
        self.W_fc1 = tf.Variable(xavier_init([7 * 7 * 64, 1024]))
        self.b_fc1 = tf.constant(0., shape=[1024])
        self.z_fc1 = tf.matmul(self.p_conv2_flatten, self.W_fc1) + self.b_fc1
        self.a_fc1 = tf.nn.relu(self.z_fc1)
        self.drop_fc1 = tf.nn.dropout(self.a_fc1, self.keep_prob)
        self.sum_his_W_fc1 = tf.summary.histogram('W_fc1', self.W_fc1)
        self.sum_his_b_fc1 = tf.summary.histogram('b_fc1', self.b_fc1)

        # fc2
        self.W_fc2 = tf.Variable(xavier_init([1024, 10]))
        self.b_fc2 = tf.constant(0., shape=[10])
        self.z_fc2 = tf.matmul(self.drop_fc1, self.W_fc2) + self.b_fc2
        self.predict = tf.nn.softmax(self.z_fc2)
        self.sum_his_W_fc2 = tf.summary.histogram('W_fc2', self.W_fc2)
        self.sum_his_b_fc2 = tf.summary.histogram('b_fc2', self.b_fc2)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(self.predict), reduction_indices=[1]))
        self.sum_sca_loss = tf.summary.scalar('loss', self.cross_entropy)
        self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.sum_sca_accu = tf.summary.scalar('accuracy', self.accuracy)

        self.sum_merge_train = tf.summary.merge([self.sum_his_W_conv1, self.sum_his_b_conv1,
                                                 self.sum_his_W_conv2, self.sum_his_b_conv2,
                                                 self.sum_his_W_fc1, self.sum_his_b_fc1,
                                                 self.sum_his_W_fc2, self.sum_his_b_fc2,
                                                 self.sum_sca_loss])
        self.sum_merge_test = tf.summary.merge([self.sum_sca_accu])
