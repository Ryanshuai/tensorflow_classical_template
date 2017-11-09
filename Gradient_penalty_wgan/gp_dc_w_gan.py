import tensorflow as tf




class GAN():
    def __init__(self):
        self.generator_reuse = False
        self.discriminator_reuse = False

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
        tensor_inputs = tf.convert_to_tensor(inputs)#[BS,W,H,D]=[BS,128,128,3]

        with tf.name_scope('d'), tf.variable_scope('d', reuse=self.discriminator_reuse):
            # conv1  #[BS,128,128,3]->[BS,64,64,64]
            W_conv1 = tf.get_variable('W_conv1', tf.truncated_normal_initializer([5, 5, 1, 64], stddev=0.02))
            b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0., shape=[64]))
            z_conv1 = tf.nn.conv2d(tensor_inputs / 255, W_conv1, strides=[1, 2, 2, 1],padding='SAME') + b_conv1
            mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
            offset_conv1 = tf.get_variable('offset_conv1',initializer=tf.zeros([64]))
            scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1,offset_conv1, scale_conv1, 1e-5)
            a_conv1 = tf.nn.leaky_relu(bn_conv1)
            sum_his_W_conv1 = tf.summary.histogram('W_conv1', W_conv1)
            sum_his_b_conv1 = tf.summary.histogram('b_conv1', b_conv1)

            # conv2  #[BS,64,64,64]->[BS,32,32,128]
            W_conv2 = tf.get_variable('W_conv2', tf.truncated_normal_initializer([5, 5, 64, 128], stddev=0.02))
            b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0., shape=[128]))
            z_conv2 = tf.nn.conv2d(a_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
            offset_conv2 = tf.get_variable('offset_conv2',initializer=tf.zeros([128]))
            scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
            a_conv2 = tf.nn.leaky_relu(bn_conv2)
            sum_his_W_conv2 = tf.summary.histogram('W_conv2', W_conv2)
            sum_his_b_conv2 = tf.summary.histogram('b_conv2', b_conv2)

            # conv3  #[BS,32,32,128]->[BS,16,16,256]
            W_conv3 = tf.get_variable('W_conv3', tf.truncated_normal_initializer([5, 5, 128, 256], stddev=0.02))
            b_conv3 = tf.get_variable('b_conv3', initializer=tf.constant(0., shape=[256]))
            z_conv3 = tf.nn.conv2d(a_conv2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, axes=[0, 1, 2])
            offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3,offset_conv3, scale_conv3, 1e-5)
            a_conv3= tf.nn.leaky_relu(bn_conv3)
            sum_his_W_conv2 = tf.summary.histogram('W_conv3', W_conv3)
            sum_his_b_conv2 = tf.summary.histogram('b_conv3', b_conv3)

            # conv4  #[BS,16,16,256]->[BS,8,8,512]
            W_conv4 = tf.get_variable('W_conv4', tf.truncated_normal_initializer([5, 5, 256, 512], stddev=0.02))
            b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0., shape=[512]))
            z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4
            mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
            offset_conv4= tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
            scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
            bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
            a_conv4= tf.nn.leaky_relu(bn_conv4)
            sum_his_W_conv2 = tf.summary.histogram('W_conv4', W_conv4)
            sum_his_b_conv2 = tf.summary.histogram('b_conv4', b_conv4)

            #classify  #[BS,8,8,512]->[BS,32768]
            batch_size = a_conv4.get_shape()[0].value
            flatten = tf.reshape(a_conv4, [batch_size, -1])
            outputs = tf.layers.dense(flatten, 2, name='outputs')


            self.discriminator_reuse = True



        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

    def build_graph(self):
        pass

    def train(self):
        pass