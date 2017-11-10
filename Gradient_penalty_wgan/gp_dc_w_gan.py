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
        tensor_rand_vec = tf.convert_to_tensor(rand_vec)  #[BS,vec_size]
        with tf.variable_scope('generator'):
            #defc
            W_defc = tf.get_variable('W_defc', initializer=tf.truncated_normal_initializer([tensor_rand_vec.shape[1].value, 4 * 4 * 1024], stddev=0.02))
            b_defc = tf.get_variable('b_defc', initializer=tf.constant_initializer([4 * 4 * 1024], 0.))
            z_defc1 = tf.matmul(tensor_rand_vec, W_defc) + b_defc
            #deflatten  # [BS,4*4*512]->[BS,4,4,512]
            deconv0 = tf.reshape(z_defc1, [-1, 4, 4, 1024])

            mean_conv0, variance_conv0 = tf.nn.moments(deconv0, axes=[0, 1, 2])
            offset_deconv0 = tf.get_variable('offset_deconv0', initializer=tf.zeros([1024]))
            scale_deconv0 = tf.get_variable('scale_deconv0', initializer=tf.ones([1024]))
            bn_deconv0 = tf.nn.batch_normalization(deconv0, mean_conv0, variance_conv0, offset_deconv0, scale_deconv0, 1e-5)
            a_deconv0 = tf.nn.relu(bn_deconv0)

            # deconv1  # [BS,4,4,1024]->[BS,8,8,512]
            W_deconv1 = tf.get_variable('W_deconv1', initializer=tf.truncated_normal_initializer([5, 5, 512, 1024], stddev=0.02))
            z_deconv1 = tf.nn.conv2d_transpose(a_deconv0, W_deconv1, [-1, 8, 8, 512], [1, 2, 2, 1])
            mean_deconv1, variance_deconv1 = tf.nn.moments(z_deconv1, axes=[0, 1, 2])
            offset_deconv1 = tf.get_variable('offset_deconv1', initializer=tf.zeros([512]))
            scale_deconv1 = tf.get_variable('scale_deconv1', initializer=tf.ones([512]))
            bn_deconv1 = tf.nn.batch_normalization(z_deconv1, mean_deconv1, variance_deconv1, offset_deconv1, scale_deconv1,1e-5)
            a_deconv1 = tf.nn.relu(bn_deconv1)

            # deconv2  # [BS,8,8,512]->[BS,16,16,256]
            W_deconv2 = tf.get_variable('W_deconv2', initializer=tf.truncated_normal_initializer([5, 5, 256, 512], stddev=0.02))
            z_deconv2 = tf.nn.conv2d_transpose(a_deconv1, W_deconv2, [-1, 16, 16, 256], [1, 2, 2, 1])
            mean_deconv2, variance_deconv2 = tf.nn.moments(z_deconv2, axes=[0, 1, 2])
            offset_deconv2 = tf.get_variable('offset_deconv2', initializer=tf.zeros([256]))
            scale_deconv2 = tf.get_variable('scale_deconv2', initializer=tf.ones([256]))
            bn_deconv2 = tf.nn.batch_normalization(z_deconv2, mean_deconv2, variance_deconv2, offset_deconv2, scale_deconv2, 1e-5)
            a_deconv2 = tf.nn.relu(bn_deconv2)

            # deconv3  # [BS,16,16,256]->[BS,32,32,128]
            W_deconv3 = tf.get_variable('W_deconv3', initializer=tf.truncated_normal_initializer([5, 5, 128, 256], stddev=0.02))
            z_deconv3 = tf.nn.conv2d_transpose(a_deconv2, W_deconv3, [-1, 32, 32, 128], [1, 2, 2, 1])
            mean_deconv3, variance_deconv3 = tf.nn.moments(z_deconv3, axes=[0, 1, 2])
            offset_deconv3 = tf.get_variable('offset_deconv3', initializer=tf.zeros([128]))
            scale_deconv3 = tf.get_variable('scale_deconv3', initializer=tf.ones([128]))
            bn_deconv3 = tf.nn.batch_normalization(z_deconv3, mean_deconv3, variance_deconv3, offset_deconv3, scale_deconv3, 1e-5)
            a_deconv3 = tf.nn.relu(bn_deconv3)

            # deconv4  # [BS,32,32,128]->[BS,64,64,64]
            W_deconv4 = tf.get_variable('W_deconv4', initializer=tf.truncated_normal_initializer([5, 5, 64, 128], stddev=0.02))
            z_deconv4 = tf.nn.conv2d_transpose(a_deconv3, W_deconv4, [-1, 64, 64, 64], [1, 2, 2, 1])
            mean_deconv4, variance_deconv4 = tf.nn.moments(z_deconv4, axes=[0, 1, 2])
            offset_deconv4 = tf.get_variable('offset_deconv4', initializer=tf.zeros([64]))
            scale_deconv4 = tf.get_variable('scale_deconv4', initializer=tf.ones([64]))
            bn_deconv4 = tf.nn.batch_normalization(z_deconv4, mean_deconv4, variance_deconv4, offset_deconv4, scale_deconv4, 1e-5)
            a_deconv4 = tf.nn.relu(bn_deconv4)

            # deconv5  # [BS,64,64,64]->[BS,128,128,3]
            W_deconv5 = tf.get_variable('W_deconv5', initializer=tf.truncated_normal_initializer([5, 5, 3, 64], stddev=0.02))
            z_deconv5 = tf.nn.conv2d_transpose(a_deconv4, W_deconv5, [-1, 128, 128, 3], [1, 2, 2, 1])
            a_deconv5 = tf.nn.tanh(z_deconv5)

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return a_deconv5

    def __discriminator(self, inputs):
        tensor_inputs = tf.convert_to_tensor(inputs)  #[BS,W,H,D]=[BS,128,128,3]
        with tf.name_scope('d'), tf.variable_scope('discriminator', reuse=self.discriminator_reuse):
            # conv1  #[BS,128,128,3]->[BS,64,64,64]
            W_conv1 = tf.get_variable('W_conv1', initializer=tf.truncated_normal_initializer([5, 5, 1, 64], stddev=0.02))
            b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant([64], 0.))
            z_conv1 = tf.nn.conv2d(tensor_inputs / 255, W_conv1, strides=[1, 2, 2, 1],padding='SAME') + b_conv1
            mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
            offset_conv1 = tf.get_variable('offset_conv1', initializer=tf.zeros([64]))
            scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1,offset_conv1, scale_conv1, 1e-5)
            a_conv1 = tf.nn.leaky_relu(bn_conv1)
            sum_his_W_conv1 = tf.summary.histogram('W_conv1', W_conv1)
            sum_his_b_conv1 = tf.summary.histogram('b_conv1', b_conv1)

            # conv2  #[BS,64,64,64]->[BS,32,32,128]
            W_conv2 = tf.get_variable('W_conv2', initializer=tf.truncated_normal_initializer([5, 5, 64, 128], stddev=0.02))
            b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant([128], 0.))
            z_conv2 = tf.nn.conv2d(a_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
            offset_conv2 = tf.get_variable('offset_conv2',initializer=tf.zeros([128]))
            scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
            a_conv2 = tf.nn.leaky_relu(bn_conv2)
            sum_his_W_conv2 = tf.summary.histogram('W_conv2', W_conv2)
            sum_his_b_conv2 = tf.summary.histogram('b_conv2', b_conv2)

            # conv3  #[BS,32,32,128]->[BS,16,16,256]
            W_conv3 = tf.get_variable('W_conv3', initializer=tf.truncated_normal_initializer([5, 5, 128, 256], stddev=0.02))
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
            W_conv4 = tf.get_variable('W_conv4', initializer=tf.truncated_normal_initializer([5, 5, 256, 512], stddev=0.02))
            b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0., shape=[512]))
            z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4
            mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
            offset_conv4= tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
            scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
            bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
            a_conv4= tf.nn.leaky_relu(bn_conv4)
            sum_his_W_conv2 = tf.summary.histogram('W_conv4', W_conv4)
            sum_his_b_conv2 = tf.summary.histogram('b_conv4', b_conv4)

            # flatten  #[BS,8,8,512]->[BS,32768]
            batch_size = a_conv4.shape[0].value
            flatten = tf.reshape(a_conv4, [batch_size, -1])

            # fc1 # classify  #[BS,32768]->[BS,1]
            W_fc1 = tf.get_variable('W_fc1', initializer=tf.truncated_normal_initializer([flatten.shape[-1], 1], stddev=0.02))
            b_fc1 = tf.get_variable('b_fc1', initializer=tf.constant_initializer([1], 0.))
            logits = tf.matmul(flatten, W_fc1) + b_fc1

        self.discriminator_reuse = True
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits

    def build_graph(self):
        pass

    def train(self):
        pass