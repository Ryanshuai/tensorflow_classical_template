import tensorflow as tf




class GAN():
    def __get_image(self):
        pass

    def __get_random_vector(self):
        pass

    def __generator(self, rand_vec):
        pass

    def __discriminator(self, inputs):
        pass

    def build_graph(self):
        pass

    def train(self):
        pass


class gp_dc_w_gan(GAN):
    def __generator(self, rand_vec):
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

    def __discriminator(self, inputs):
        outputs = tf.convert_to_tensor(inputs)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # conv1
            self.W_conv1 = tf.get_variable(xavier_init_conv2d([5, 5, 1, 64]))
            self.b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0., shape=[64]))
            self.z_conv1 = tf.nn.conv2d(self.input / 255, self.W_conv1, strides=[1, 2, 2, 1],padding='SAME') + self.b_conv1
            self.mean_conv1, self.variance_conv1 = tf.nn.moments(self.z_conv1, axes=[0, 1, 2])
            self.offset_conv1 = tf.get_variable('offset_conv1',initializer=tf.zeros([64]))
            self.scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            self.bn_conv1 = tf.nn.batch_normalization(self.z_conv1, self.mean_conv1, self.variance_conv1,self.offset_conv1, self.scale_conv1, 0.001)
            self.a_conv1 = tf.nn.leaky_relu(self.bn_conv1)
            self.sum_his_W_conv1 = tf.summary.histogram('W_conv1', self.W_conv1)
            self.sum_his_b_conv1 = tf.summary.histogram('b_conv1', self.b_conv1)

            # conv2
            self.W_conv2 = tf.get_variable(xavier_init_conv2d([5, 5, 64, 128]))
            self.b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0., shape=[128]))
            self.z_conv2 = tf.nn.conv2d(self.a_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv2
            self.mean_conv2, self.variance_conv2 = tf.nn.moments(self.z_conv2, axes=[0, 1, 2])
            self.offset_conv2 = tf.get_variable('offset_conv2',initializer=tf.zeros([128]))
            self.scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            self.bn_conv2 = tf.nn.batch_normalization(self.z_conv2, self.mean_conv2, self.variance_conv2, self.offset_conv2, self.scale_conv2, 0.001)
            self.a_conv2 = tf.nn.leaky_relu(self.bn_conv2)
            self.sum_his_W_conv2 = tf.summary.histogram('W_conv2', self.W_conv2)
            self.sum_his_b_conv2 = tf.summary.histogram('b_conv2', self.b_conv2)

            # conv3
            self.W_conv3 = tf.get_variable(xavier_init_conv2d([5, 5, 128, 256]))
            self.b_conv3 = tf.get_variable('b_conv2', initializer=tf.constant(0., shape=[256]))
            self.z_conv3 = tf.nn.conv2d(self.a_conv2, self.W_conv3, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv3
            self.mean_conv3, self.variance_conv3 = tf.nn.moments(self.z_conv2, axes=[0, 1, 2])
            self.offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            self.scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            self.bn_conv3 = tf.nn.batch_normalization(self.z_conv3, self.mean_conv3, self.variance_conv3,self.offset_conv3, self.scale_conv3, 0.001)
            self.a_conv3= tf.nn.leaky_relu(self.bn_conv2)
            self.sum_his_W_conv2 = tf.summary.histogram('W_conv2', self.W_conv2)
            self.sum_his_b_conv2 = tf.summary.histogram('b_conv2', self.b_conv2)

            # conv4
            self.W_conv4 = tf.get_variable(xavier_init_conv2d([5, 5, 128, 256]))
            self.b_conv4 = tf.get_variable('b_conv2', initializer=tf.constant(0., shape=[256]))
            self.z_conv4 = tf.nn.conv2d(self.a_conv2, self.W_conv3, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv3
            self.mean_conv4, self.variance_conv3 = tf.nn.moments(self.z_conv2, axes=[0, 1, 2])
            self.offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            self.scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            self.bn_conv3 = tf.nn.batch_normalization(self.z_conv3, self.mean_conv3, self.variance_conv3,self.offset_conv3, self.scale_conv3, 0.001)
            self.a_conv3= tf.nn.leaky_relu(self.bn_conv2)
            self.sum_his_W_conv2 = tf.summary.histogram('W_conv2', self.W_conv2)
            self.sum_his_b_conv2 = tf.summary.histogram('b_conv2', self.b_conv2)


            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

    def build_graph(self):
        pass

    def train(self):
        pass