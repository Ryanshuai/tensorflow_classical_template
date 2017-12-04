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
        # flatten  #[BS,z_dim]->[BS,784]
        flatten = tf.reshape(tensor_inputs, [self.BS, -1])

        # fc1 #[BS,784]->[BS,400]
        W_fc1 = tf.get_variable('W_fc1', [flatten.shape[1].value, 400],initializer=tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.get_variable('b_fc1', [400], initializer=tf.constant_initializer(0.))
        z_fc1 = tf.matmul(flatten, W_fc1) + b_fc1
        mean_fc1, variance_fc1 = tf.nn.moments(z_fc1, axes=[0])
        offset_fc1 = tf.get_variable('offset_fc1', initializer=tf.zeros([400]))
        scale_fc1 = tf.get_variable('scale_fc1', initializer=tf.ones([400]))
        bn_fc1 = tf.nn.batch_normalization(z_fc1, mean_fc1, variance_fc1, offset_fc1, scale_fc1, 1e-9)
        a_fc1 = tf.nn.relu(bn_fc1)

        # fc1 #[BS,400]->[BS,2*z_dim]
        W_fc2 = tf.get_variable('W_fc2', [400, 2 * self.z_dim],initializer=tf.contrib.layers.xavier_initializer())
        b_fc2 = tf.get_variable('b_fc2', [2 * self.z_dim], initializer=tf.constant_initializer(0.))
        z_fc2 = tf.matmul(a_fc1, W_fc2) + b_fc2
        mean_fc2, variance_fc2 = tf.nn.moments(z_fc2, axes=[0])
        offset_fc2 = tf.get_variable('offset_fc2', initializer=tf.zeros([2 * self.z_dim]))
        scale_fc2 = tf.get_variable('scale_fc2', initializer=tf.ones([2 * self.z_dim]))
        bn_fc2 = tf.nn.batch_normalization(z_fc2, mean_fc2, variance_fc2, offset_fc2, scale_fc2, 1e-9)
        a_fc2 = tf.nn.relu(bn_fc2)

        mean, stddev = tf.split(a_fc2, 2, axis=1)  # [bs, z_dim],#[bs, z_dim]
        stddev = 1e-6 + tf.nn.softplus(stddev) # [bs, z_dim]
        return mean, stddev # [bs, z_dim],#[bs, z_dim]


    def _decoder(self, z):
        tensor_z = tf.convert_to_tensor(z)  # [BS,vec_size]
        # defc1 #[BS,2*z_dim]->[BS,400]
        W_defc1 = tf.get_variable('W_defc1', [self.z_dim, 400], initializer=tf.contrib.layers.xavier_initializer())
        b_defc1 = tf.get_variable('b_defc1', [400], initializer=tf.constant_initializer(0.))
        z_defc1 = tf.matmul(tensor_z, W_defc1) + b_defc1
        mean_defc1, variance_defc1 = tf.nn.moments(z_defc1, axes=[0])
        offset_defc1 = tf.get_variable('offset_defc1', initializer=tf.zeros([400]))
        scale_defc1 = tf.get_variable('scale_defc1', initializer=tf.ones([400]))
        bn_defc1 = tf.nn.batch_normalization(z_defc1, mean_defc1, variance_defc1, offset_defc1, scale_defc1, 1e-9)
        a_defc1 = tf.nn.relu(bn_defc1)

        # defc2 #[BS,400]->[BS,784]
        W_defc2 = tf.get_variable('W_defc2', [400, 784], initializer=tf.contrib.layers.xavier_initializer())
        b_defc2 = tf.get_variable('b_defc2', [784], initializer=tf.constant_initializer(0.))
        z_defc2 = tf.matmul(a_defc1, W_defc2) + b_defc2
        mean_defc2, variance_defc2 = tf.nn.moments(z_defc2, axes=[0])
        offset_defc2 = tf.get_variable('offset_defc2', initializer=tf.zeros([784]))
        scale_defc2 = tf.get_variable('scale_defc2', initializer=tf.ones([784]))
        bn_defc2 = tf.nn.batch_normalization(z_defc2, mean_defc2, variance_defc2, offset_defc2, scale_defc2, 1e-9)
        bn_defc2_reshape = tf.reshape(bn_defc2, [self.BS, 28, 28 ,1])

        return bn_defc2_reshape #recon_X


    def build_graph(self):
        tf.reset_default_graph()
        # placeholder
        self.X = tf.placeholder(tf.float32, [self.BS, 784], name='real_images') #[BS,784]
        self.X_reshape = tf.reshape(self.X, [self.BS, 28, 28, 1]) #[BS,W,H,C]

        # VAE
        self.mu, self.sigma = self._encoder(self.X_reshape) # [bs,z_dim],#[bs,z_dim]
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32) #[bs, z_dim]
        self.recon_X = self._decoder(self.z)
        # loss
        IO_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.recon_X, labels=self.X_reshape),[1, 2, 3])# [bs,w,h,c]->[bs,1]
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