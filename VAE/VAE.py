import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

import prior_factory as prior

class VAE(object):
    model_name = "VAE"     # name for checkpoint

    # Gaussian Encoder
    def _encoder(self, x):
        with tf.variable_scope("encoder", reuse=reuse):

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

        return mean, stddev


    def _decoder(self, z):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("decoder", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
                   scope='de_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='de_dc4'))
            return out


    def build_model(self):
        # placeholder
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images') #[BS,W,H,C]
        # VAE
        mu, sigma = self._encoder(self.inputs)
        z = mu + sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32) #[BS,......]
        out = self._decoder(z)
        self.out = tf.clip_by_value(out, 1e-8, 1 - 1e-8)
        # loss
        IO_loss = tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out))
        KL_loss = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1)

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - self.KL_divergence

        self.loss = -ELBO

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        self.fake_images = self.decoder(self.z, is_training=False, reuse=True)

        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = prior.gaussian(self.batch_size, self.z_dim)

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.neg_loglikelihood, self.KL_divergence],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, nll_loss, kl_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = prior.gaussian(self.batch_size, self.z_dim)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ learned manifold """
        if self.z_dim == 2:
            assert self.z_dim == 2

            z_tot = None
            id_tot = None
            for idx in range(0, 100):
                #randomly sampling
                id = np.random.randint(0,self.num_batches)
                batch_images = self.data_X[id * self.batch_size:(id + 1) * self.batch_size]
                batch_labels = self.data_y[id * self.batch_size:(id + 1) * self.batch_size]

                z = self.sess.run(self.mu, feed_dict={self.inputs: batch_images})

                if idx == 0:
                    z_tot = z
                    id_tot = batch_labels
                else:
                    z_tot = np.concatenate((z_tot, z), axis=0)
                    id_tot = np.concatenate((id_tot, batch_labels), axis=0)

            save_scattered_image(z_tot, id_tot, -4, 4, name=check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_learned_manifold.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


class VAE():
    def __init__(self, batch_size):
        self._BS = batch_size

    def _get_dataset(self):
        #get image paths
        current_dir = os.getcwd()
        # parent = os.path.dirname(current_dir)
        pokemon_dir = os.path.join(current_dir, 'real_image')
        image_paths = []
        for each in os.listdir(pokemon_dir):
            image_paths.append(os.path.join(pokemon_dir, each))
        tensor_image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        #data processing func used in map
        def preprocessing(filename):
            image_string = tf.read_file(filename)
            image = tf.image.decode_png(image_string)
            image = tf.image.resize_images(image, [128, 128])
            image.set_shape([128, 128, 3])
            # image = tf.image.random_flip_left_right(image)
            # image = tf.image.random_brightness(image, max_delta=0.1)
            # image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            return image
        #make dataset
        dataset = tf.data.Dataset.from_tensor_slices(tensor_image_paths)
        #dataset = dataset.repeat(32)
        dataset = dataset.map(preprocessing)
        # dataset = dataset.shuffle(3200)
        dataset = dataset.batch(self._BS)

        return dataset


    def _generator(self, rand_vec):
        pass

    def _discriminator(self, inputs):
        pass

    def build_graph(self):
        # placeholder
        self.random_vec = tf.placeholder(tf.float32, shape=[None, 100])
        self.real_image = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        # gan
        self.fake_image = self._generator(self.random_vec) #[BS,128,128,3]
        real_logits = self._discriminator(self.real_image) #[BS,1]
        fake_logits = self._discriminator(self.fake_image) #[BS,1]
        # loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits))) #[1]
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits))) #[1]
        self.d_loss = d_loss_real + d_loss_fake #[1]

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits))) #[1]

        self.sum_his_d_loss = tf.summary.scalar('d_loss', self.d_loss)
        self.sum_his_g_loss = tf.summary.scalar('g_loss', self.g_loss)
        self.sum_merge = tf.summary.merge_all()
        # optimize
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_variables)
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_variables)


    def train(self):
        tf_sum_writer = tf.summary.FileWriter('logs')

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='tfModel/')

        image_dataset = self._get_dataset()
        iterator = image_dataset.make_initializable_iterator()

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
                sess.run(iterator.initializer)
                epoch_step = 0
                while True:
                    try:
                        real_image = sess.run(iterator.get_next())
                    except tf.errors.OutOfRangeError:
                        break
                    rand_vec = self._get_random_vector()
                    _, dLoss, sum_d = sess.run([self.d_optimizer, self.d_loss, self.sum_his_d_loss],
                                        feed_dict={self.random_vec: rand_vec, self.real_image: real_image})
                    _, gLoss, sum_g = sess.run([self.g_optimizer, self.g_loss, self.sum_his_g_loss],
                                        feed_dict={self.random_vec: rand_vec})

                    if epoch_step % 10 == 0: # tensorboard
                        tf_sum_writer.add_summary(sum_d, global_step=global_step)
                        tf_sum_writer.add_summary(sum_g, global_step=global_step)

                    print('epoch:', epoch, 'epoch_step:', epoch_step, 'global_step:', global_step)
                    epoch_step, global_step = epoch_step + 1, global_step + 1



                if epoch % 5 == 0: # save model
                    print('---------------------')
                    if not os.path.exists('./tfModel/'):
                        os.makedirs('./tfModel/')
                    saver.save(sess, './tfModel/epoch' + str(epoch))

                if epoch % 1 == 0: # save images
                    fake_image_path = './generated_image/epoch' + str(epoch)
                    if not os.path.exists(fake_image_path):
                        os.makedirs(fake_image_path)
                    test_vec = self._get_random_vector()
                    img_test = sess.run(self.fake_image, feed_dict={self.random_vec: test_vec})
                    img_test = img_test * 255.0
                    img_test.astype(np.uint8)
                    for i in range(self._BS):
                        cv2.imwrite(fake_image_path + '/' + str(i) + '.jpg', img_test[i])