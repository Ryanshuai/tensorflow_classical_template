import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data


class VAE(object):
    def __init__(self, batch_size, z_dim):
        self.BS = batch_size
        self.z_dim = z_dim


    def _get_random_vector(self, mu=None, sigma=None): #mu:[z_dim], #sigma:[z_dim]
        if(mu):
            return np.random.normal(loc=mu, scale=sigma, size=[self.BS, self.z_dim]).astype(np.float32)
        else:
            return np.random.normal(size=[self.BS, self.z_dim]).astype(np.float32)


    def _get_dataset(self):
        return input_data.read_data_sets('MNIST_data', one_hot=True)


    def _encoder(self, X):
        tensor_inputs = tf.convert_to_tensor(X)  # [BS,W,H,D]=[BS,128,128,3]

        # conv1  #[BS,128,128,3]->[BS,64,64,64]
        W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0.))
        z_conv1 = tf.nn.conv2d(tensor_inputs / 255, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
        offset_conv1 = tf.get_variable('offset_conv1', initializer=tf.zeros([64]))
        scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
        bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1, offset_conv1, scale_conv1, 1e-5)
        a_conv1 = tf.nn.leaky_relu(bn_conv1)

        # conv2  #[BS,64,64,64]->[BS,32,32,128]
        W_conv2 = tf.get_variable('W_conv2', [5, 5, 64, 128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0.))
        z_conv2 = tf.nn.conv2d(a_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
        offset_conv2 = tf.get_variable('offset_conv2', initializer=tf.zeros([128]))
        scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
        bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
        a_conv2 = tf.nn.leaky_relu(bn_conv2)

        # conv3  #[BS,32,32,128]->[BS,16,16,256]
        W_conv3 = tf.get_variable('W_conv3', [5, 5, 128, 256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv3 = tf.get_variable('b_conv3', initializer=tf.constant(0.))
        z_conv3 = tf.nn.conv2d(a_conv2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, axes=[0, 1, 2])
        offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
        scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
        bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3, offset_conv3, scale_conv3, 1e-5)
        a_conv3 = tf.nn.leaky_relu(bn_conv3)

        # conv4  #[BS,16,16,256]->[BS,8,8,512]
        W_conv4 = tf.get_variable('W_conv4', [5, 5, 256, 512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0.))
        z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4
        mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
        offset_conv4 = tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
        scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
        bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
        a_conv4 = tf.nn.leaky_relu(bn_conv4)

        # flatten  #[BS,8,8,512]->[BS,32768]
        flatten = tf.reshape(a_conv4, [self.BS, -1])

        # fc1 #[BS,32768]->[BS,1024]
        W_fc1 = tf.get_variable('W_fc1', [flatten.shape[1].value, 1024],initializer=tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.get_variable('b_fc1', [1024], initializer=tf.constant_initializer(0.))
        z_fc1 = tf.matmul(flatten, W_fc1) + b_fc1
        mean_fc1, variance_fc1 = tf.nn.moments(z_fc1, axes=[0])
        offset_fc1 = tf.get_variable('offset_fc1', initializer=tf.zeros([1024]))
        scale_fc1 = tf.get_variable('scale_fc1', initializer=tf.ones([1024]))
        bn_fc1 = tf.nn.batch_normalization(z_fc1, mean_fc1, variance_fc1, offset_fc1, scale_fc1, 0.001)
        a_fc1 = tf.nn.leaky_relu(bn_fc1)

        # fc1 #[BS,1024]->[BS,2*z_dim]
        W_fc2 = tf.get_variable('W_fc2', [1024, 2 * self.z_dim],initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.get_variable('b_fc2', [2 * self.z_dim], initializer=tf.constant_initializer(0.))
        z_fc2 = tf.matmul(a_fc1, W_fc2) + b_fc2
        mean_fc2, variance_fc2 = tf.nn.moments(z_fc2, axes=[0])
        offset_fc2 = tf.get_variable('offset_fc2', initializer=tf.zeros([2 * self.z_dim]))
        scale_fc2 = tf.get_variable('scale_fc2', initializer=tf.ones([2 * self.z_dim]))
        bn_fc2 = tf.nn.batch_normalization(z_fc2, mean_fc2, variance_fc2, offset_fc2, scale_fc2, 0.001)
        a_fc2 = tf.nn.leaky_relu(bn_fc2)

        mean, stddev = tf.split(a_fc2, 2, axis=1)  # [bs, z_dim],#[bs, z_dim]
        stddev = 1e-6 + tf.nn.softplus(stddev) # [bs, z_dim]
        return mean, stddev # [bs, z_dim],#[bs, z_dim]


    def _decoder(self, z):
        tensor_z = tf.convert_to_tensor(z)  # [BS,vec_size]
        # defc
        W_defc = tf.get_variable('W_defc', [tensor_z.shape[1].value, 4 * 4 * 1024],initializer=tf.contrib.layers.xavier_initializer())
        b_defc = tf.get_variable('b_defc', [4 * 4 * 1024], initializer=tf.constant_initializer(0.))
        z_defc1 = tf.matmul(tensor_z, W_defc) + b_defc
        # deflatten  # [BS,4*4*512]->[BS,4,4,512]
        deconv0 = tf.reshape(z_defc1, [-1, 4, 4, 1024])

        mean_conv0, variance_conv0 = tf.nn.moments(deconv0, axes=[0, 1, 2])
        offset_deconv0 = tf.get_variable('offset_deconv0', initializer=tf.zeros([1024]))
        scale_deconv0 = tf.get_variable('scale_deconv0', initializer=tf.ones([1024]))
        bn_deconv0 = tf.nn.batch_normalization(deconv0, mean_conv0, variance_conv0, offset_deconv0, scale_deconv0, 1e-5)
        a_deconv0 = tf.nn.relu(bn_deconv0)

        # deconv1  # [BS,4,4,1024]->[BS,8,8,512]
        W_deconv1 = tf.get_variable('W_deconv1', [5, 5, 512, 1024],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        z_deconv1 = tf.nn.conv2d_transpose(a_deconv0, W_deconv1, [self.BS, 8, 8, 512], [1, 2, 2, 1])
        mean_deconv1, variance_deconv1 = tf.nn.moments(z_deconv1, axes=[0, 1, 2])
        offset_deconv1 = tf.get_variable('offset_deconv1', initializer=tf.zeros([512]))
        scale_deconv1 = tf.get_variable('scale_deconv1', initializer=tf.ones([512]))
        bn_deconv1 = tf.nn.batch_normalization(z_deconv1, mean_deconv1, variance_deconv1, offset_deconv1, scale_deconv1,1e-5)
        a_deconv1 = tf.nn.relu(bn_deconv1)

        # deconv2  # [BS,8,8,512]->[BS,16,16,256]
        W_deconv2 = tf.get_variable('W_deconv2', [5, 5, 256, 512],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        z_deconv2 = tf.nn.conv2d_transpose(a_deconv1, W_deconv2, [self.BS, 16, 16, 256], [1, 2, 2, 1])
        mean_deconv2, variance_deconv2 = tf.nn.moments(z_deconv2, axes=[0, 1, 2])
        offset_deconv2 = tf.get_variable('offset_deconv2', initializer=tf.zeros([256]))
        scale_deconv2 = tf.get_variable('scale_deconv2', initializer=tf.ones([256]))
        bn_deconv2 = tf.nn.batch_normalization(z_deconv2, mean_deconv2, variance_deconv2, offset_deconv2, scale_deconv2,1e-5)
        a_deconv2 = tf.nn.relu(bn_deconv2)

        # deconv3  # [BS,16,16,256]->[BS,32,32,128]
        W_deconv3 = tf.get_variable('W_deconv3', [5, 5, 128, 256],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        z_deconv3 = tf.nn.conv2d_transpose(a_deconv2, W_deconv3, [self.BS, 32, 32, 128], [1, 2, 2, 1])
        mean_deconv3, variance_deconv3 = tf.nn.moments(z_deconv3, axes=[0, 1, 2])
        offset_deconv3 = tf.get_variable('offset_deconv3', initializer=tf.zeros([128]))
        scale_deconv3 = tf.get_variable('scale_deconv3', initializer=tf.ones([128]))
        bn_deconv3 = tf.nn.batch_normalization(z_deconv3, mean_deconv3, variance_deconv3, offset_deconv3, scale_deconv3,1e-5)
        a_deconv3 = tf.nn.relu(bn_deconv3)

        # deconv4  # [BS,32,32,128]->[BS,64,64,64]
        W_deconv4 = tf.get_variable('W_deconv4', [5, 5, 64, 128],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        z_deconv4 = tf.nn.conv2d_transpose(a_deconv3, W_deconv4, [self.BS, 64, 64, 64], [1, 2, 2, 1])
        mean_deconv4, variance_deconv4 = tf.nn.moments(z_deconv4, axes=[0, 1, 2])
        offset_deconv4 = tf.get_variable('offset_deconv4', initializer=tf.zeros([64]))
        scale_deconv4 = tf.get_variable('scale_deconv4', initializer=tf.ones([64]))
        bn_deconv4 = tf.nn.batch_normalization(z_deconv4, mean_deconv4, variance_deconv4, offset_deconv4, scale_deconv4,1e-5)
        a_deconv4 = tf.nn.relu(bn_deconv4)

        # deconv5  # [BS,64,64,64]->[BS,128,128,3]
        W_deconv5 = tf.get_variable('W_deconv5', [5, 5, 3, 64],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        z_deconv5 = tf.nn.conv2d_transpose(a_deconv4, W_deconv5, [self.BS, 128, 128, 3], [1, 2, 2, 1])

        return z_deconv5 #recon_X


    def build_graph(self):
        tf.reset_default_graph()
        # placeholder
        self.X = tf.placeholder(tf.float32, [self.BS, 28, 28 ,3], name='real_images') #[BS,W,H,C]
        self._X = tf.image.resize_bicubic(self.X, [128, 128])
        # VAE
        self.mu, self.sigma = self._encoder(self._X) # [bs, z_dim],#[bs, z_dim]
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32) #[bs, z_dim]
        self.recon_X = self._decoder(self.z)
        # loss
        IO_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.recon_X, labels=self._X),[1, 2, 3])# [bs,w,h,c]->[bs,1]
        KL_loss = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1,[1])# [bs,z_dim]->[bs,1]
        self.IO_loss = tf.reduce_mean(IO_loss)
        self.KL_loss = tf.reduce_mean(KL_loss)
        self.loss = self.IO_loss + self.KL_loss
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        #tensorboard
        self.sum_IO_loss = tf.summary.scalar("IO_loss", self.IO_loss)
        self.sum_KL_loss = tf.summary.scalar("KL_loss", self.KL_loss)
        self.sum_loss = tf.summary.scalar("loss", self.loss)
        self.sum_merge = tf.summary.merge_all()


    def train(self):
        tf_sum_writer = tf.summary.FileWriter('logs')

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='tfModel/')

        mnist = self._get_dataset()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if ckpt and ckpt.model_checkpoint_path:
                print('loading_model')
                saver.restore(sess, ckpt.model_checkpoint_path)
                pre_model_epoch = int(ckpt.model_checkpoint_path[13:])
                print('pre_model_epoch:', pre_model_epoch)
            else:
                pre_model_epoch = 0
                print('no_pre_model')

            tf_sum_writer.add_graph(sess.graph)
            global_step = 0
            for epoch in range(pre_model_epoch + 1, pre_model_epoch + 500):
                for epoch_step in range(150):
                    X, label = mnist.train.next_batch(self.BS)
                    X = np.reshape(X, [self.BS, 28, 28, 1])
                    X = np.concatenate((X, X, X), axis=3)
                    #train
                    _, sum_merge, loss, IO_loss, KL_loss = sess.run(
                        [self.optimizer, self.sum_merge, self.loss, self.IO_loss, self.KL_loss],
                        feed_dict={self.X: X})

                    if epoch_step % 10 == 0: # tensorboard
                        tf_sum_writer.add_summary(sum_merge, global_step=global_step)

                    print('epoch:', epoch, 'epoch_step:', epoch_step, 'global_step:', global_step)
                    global_step = global_step + 1

                if epoch % 50 == 0: # save model
                    print('---------------------')
                    if not os.path.exists('./tfModel/'):
                        os.makedirs('./tfModel/')
                    saver.save(sess, './tfModel/epoch' + str(epoch))

                if epoch % 10 == 0: # save images
                    generated_image = './generated_image/epoch' + str(epoch)
                    if not os.path.exists(generated_image):
                        os.makedirs(generated_image)
                    test_vec = self._get_random_vector()
                    img_test = sess.run(self.recon_X, feed_dict={self.z: test_vec})
                    img_test = img_test * 255.0
                    img_test.astype(np.uint8)
                    for i in range(self.BS):
                        cv2.imwrite(generated_image + '/' + str(i) + '.jpg', img_test[i])


if __name__ == "__main__":
    vae = VAE(32, 10)
    vae.build_graph()
    vae.train()