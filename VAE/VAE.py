import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

import prior_factory as prior

class VAE(object):
    def __init__(self, batch_size, z_dim):
        self._BS = batch_size
        self.z_dim = z_dim


    def _get_random_vector(self, mu=None, sigma=None):
        if(mu):
            pass
        else:
            return np.random.normal(size=[self._BS, self.z_dim]).astype(np.float32)


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


    def _encoder(self, x):
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
        net = tf.reshape(net, [self.batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
        gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')




        mean, stddev = tf.split(gaussian_params, 2, axis=1)  # [bs, z_dim],#[bs, z_dim]
        stddev = 1e-6 + tf.nn.softplus(stddev) # [bs, z_dim]
        return mean, stddev # [bs, z_dim],#[bs, z_dim]


    def _decoder(self, z):
        net = tf.nn.relu(bn(linear(z, 1024, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
        net = tf.reshape(net, [self.batch_size, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='de_dc3'), is_training=is_training,
               scope='de_bn3'))



        out = deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='de_dc4')
        return out


    def build_model(self):
        # placeholder
        self.X = tf.placeholder(tf.float32, [self.BS, 28, 28 ,1], name='real_images') #[BS,W,H,C]
        # VAE
        self.mu, self.sigma = self._encoder(self.X) # [bs, z_dim],#[bs, z_dim]
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32) #[bs, z_dim]
        self.out = self._decoder(z)
        clip_out = tf.clip_by_value(self.out, 1e-8, 1 - 1e-8)
        # loss
        self.IO_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=clip_out, labels=self.X),[1, 2, 3])
        self.KL_loss = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1,[1])
        self.loss = tf.reduce_mean(self.IO_loss + self.KL_loss)
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(self.loss)
        #tensorboard
        self.sum_IO_loss = tf.summary.scalar("IO_loss", self.IO_loss)
        self.sum_KL_loss = tf.summary.scalar("KL_loss", self.KL_loss)
        self.sum_loss = tf.summary.scalar("loss", self.loss)
        self.sum_merge = tf.summary.merge_all()


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
                        X = sess.run(iterator.get_next())
                    except tf.errors.OutOfRangeError:
                        break
                    #train
                    _, sum_merge, loss, IO_loss, KL_loss = sess.run(
                        [self.optimizer, self.sum_merge, self.loss, self.IO_loss, self.KL_loss],
                        feed_dict={self.X: X})

                    if epoch_step % 10 == 0: # tensorboard
                        tf_sum_writer.add_summary(sum_merge, global_step=global_step)

                    print('epoch:', epoch, 'epoch_step:', epoch_step, 'global_step:', global_step)
                    epoch_step, global_step = epoch_step + 1, global_step + 1

                if epoch % 50 == 0: # save model
                    print('---------------------')
                    if not os.path.exists('./tfModel/'):
                        os.makedirs('./tfModel/')
                    saver.save(sess, './tfModel/epoch' + str(epoch))

                if epoch % 10 == 0: # save images
                    fake_image_path = './generated_image/epoch' + str(epoch)
                    if not os.path.exists(fake_image_path):
                        os.makedirs(fake_image_path)
                    test_vec = self._get_random_vector()
                    img_test = sess.run(self.fake_image, feed_dict={self.z: test_vec})
                    img_test = img_test * 255.0
                    img_test.astype(np.uint8)
                    for i in range(self._BS):
                        cv2.imwrite(fake_image_path + '/' + str(i) + '.jpg', img_test[i])


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

