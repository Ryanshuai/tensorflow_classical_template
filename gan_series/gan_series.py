import tensorflow as tf
import os
import numpy as np
import cv2


class GAN():
    def __init__(self, batch_size, z_dim):
        self._BS = batch_size
        self.z_dim = z_dim
        self._discriminator_reuse = False

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

    def _get_random_vector(self):
        rand_vec = np.random.normal(size=[self._BS, self.z_dim]).astype(np.float32)
        return rand_vec

    def _generator(self, rand_vec):
        pass

    def _discriminator(self, inputs):
        pass

    def build_graph(self):
        # placeholder
        self.random_vec = tf.placeholder(tf.float32, shape=[None, self.z_dim])
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


class dc_w_gan(GAN):
    def _generator(self, rand_vec):
        tensor_rand_vec = tf.convert_to_tensor(rand_vec)  #[BS,vec_size]
        with tf.variable_scope('generator'):
            #defc
            W_defc = tf.get_variable('W_defc', [tensor_rand_vec.shape[1].value, 4 * 4 * 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_defc = tf.get_variable('b_defc', [4 * 4 * 1024], initializer=tf.constant_initializer(0.))
            z_defc1 = tf.matmul(tensor_rand_vec, W_defc) + b_defc
            #deflatten  # [BS,4*4*512]->[BS,4,4,512]
            deconv0 = tf.reshape(z_defc1, [-1, 4, 4, 1024])

            mean_conv0, variance_conv0 = tf.nn.moments(deconv0, axes=[0, 1, 2])
            offset_deconv0 = tf.get_variable('offset_deconv0', initializer=tf.zeros([1024]))
            scale_deconv0 = tf.get_variable('scale_deconv0', initializer=tf.ones([1024]))
            bn_deconv0 = tf.nn.batch_normalization(deconv0, mean_conv0, variance_conv0, offset_deconv0, scale_deconv0, 1e-5)
            a_deconv0 = tf.nn.relu(bn_deconv0)

            # deconv1  # [BS,4,4,1024]->[BS,8,8,512]
            W_deconv1 = tf.get_variable('W_deconv1', [5, 5, 512, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv1 = tf.nn.conv2d_transpose(a_deconv0, W_deconv1, [self._BS, 8, 8, 512], [1, 2, 2, 1])
            mean_deconv1, variance_deconv1 = tf.nn.moments(z_deconv1, axes=[0, 1, 2])
            offset_deconv1 = tf.get_variable('offset_deconv1', initializer=tf.zeros([512]))
            scale_deconv1 = tf.get_variable('scale_deconv1', initializer=tf.ones([512]))
            bn_deconv1 = tf.nn.batch_normalization(z_deconv1, mean_deconv1, variance_deconv1, offset_deconv1, scale_deconv1,1e-5)
            a_deconv1 = tf.nn.relu(bn_deconv1)

            # deconv2  # [BS,8,8,512]->[BS,16,16,256]
            W_deconv2 = tf.get_variable('W_deconv2', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv2 = tf.nn.conv2d_transpose(a_deconv1, W_deconv2, [self._BS, 16, 16, 256], [1, 2, 2, 1])
            mean_deconv2, variance_deconv2 = tf.nn.moments(z_deconv2, axes=[0, 1, 2])
            offset_deconv2 = tf.get_variable('offset_deconv2', initializer=tf.zeros([256]))
            scale_deconv2 = tf.get_variable('scale_deconv2', initializer=tf.ones([256]))
            bn_deconv2 = tf.nn.batch_normalization(z_deconv2, mean_deconv2, variance_deconv2, offset_deconv2, scale_deconv2, 1e-5)
            a_deconv2 = tf.nn.relu(bn_deconv2)

            # deconv3  # [BS,16,16,256]->[BS,32,32,128]
            W_deconv3 = tf.get_variable('W_deconv3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv3 = tf.nn.conv2d_transpose(a_deconv2, W_deconv3, [self._BS, 32, 32, 128], [1, 2, 2, 1])
            mean_deconv3, variance_deconv3 = tf.nn.moments(z_deconv3, axes=[0, 1, 2])
            offset_deconv3 = tf.get_variable('offset_deconv3', initializer=tf.zeros([128]))
            scale_deconv3 = tf.get_variable('scale_deconv3', initializer=tf.ones([128]))
            bn_deconv3 = tf.nn.batch_normalization(z_deconv3, mean_deconv3, variance_deconv3, offset_deconv3, scale_deconv3, 1e-5)
            a_deconv3 = tf.nn.relu(bn_deconv3)

            # deconv4  # [BS,32,32,128]->[BS,64,64,64]
            W_deconv4 = tf.get_variable('W_deconv4', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv4 = tf.nn.conv2d_transpose(a_deconv3, W_deconv4, [self._BS, 64, 64, 64], [1, 2, 2, 1])
            mean_deconv4, variance_deconv4 = tf.nn.moments(z_deconv4, axes=[0, 1, 2])
            offset_deconv4 = tf.get_variable('offset_deconv4', initializer=tf.zeros([64]))
            scale_deconv4 = tf.get_variable('scale_deconv4', initializer=tf.ones([64]))
            bn_deconv4 = tf.nn.batch_normalization(z_deconv4, mean_deconv4, variance_deconv4, offset_deconv4, scale_deconv4, 1e-5)
            a_deconv4 = tf.nn.relu(bn_deconv4)

            # deconv5  # [BS,64,64,64]->[BS,128,128,3]
            W_deconv5 = tf.get_variable('W_deconv5', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv5 = tf.nn.conv2d_transpose(a_deconv4, W_deconv5, [self._BS, 128, 128, 3], [1, 2, 2, 1])
            a_deconv5 = tf.nn.tanh(z_deconv5)
            self.a_deconv5 = a_deconv5

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return a_deconv5

    def _discriminator(self, inputs):
        tensor_inputs = tf.convert_to_tensor(inputs)  #[BS,W,H,D]=[BS,128,128,3]
        with tf.name_scope('d'), tf.variable_scope('discriminator', reuse=self._discriminator_reuse):
            # conv1  #[BS,128,128,3]->[BS,64,64,64]
            W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0.))
            z_conv1 = tf.nn.conv2d(tensor_inputs / 255, W_conv1, strides=[1, 2, 2, 1],padding='SAME') + b_conv1
            mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
            offset_conv1 = tf.get_variable('offset_conv1', initializer=tf.zeros([64]))
            scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1,offset_conv1, scale_conv1, 1e-5)
            a_conv1 = tf.nn.leaky_relu(bn_conv1)
            sum_his_W_conv1 = tf.summary.histogram('W_conv1', W_conv1)
            sum_his_b_conv1 = tf.summary.histogram('b_conv1', b_conv1)

            # conv2  #[BS,64,64,64]->[BS,32,32,128]
            W_conv2 = tf.get_variable('W_conv2', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0.))
            z_conv2 = tf.nn.conv2d(a_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
            offset_conv2 = tf.get_variable('offset_conv2',initializer=tf.zeros([128]))
            scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
            a_conv2 = tf.nn.leaky_relu(bn_conv2)
            sum_his_W_conv2 = tf.summary.histogram('W_conv2', W_conv2)
            sum_his_b_conv2 = tf.summary.histogram('b_conv2', b_conv2)

            # conv3  #[BS,32,32,128]->[BS,16,16,256]
            W_conv3 = tf.get_variable('W_conv3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv3 = tf.get_variable('b_conv3', initializer=tf.constant(0.))
            z_conv3 = tf.nn.conv2d(a_conv2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, axes=[0, 1, 2])
            offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3,offset_conv3, scale_conv3, 1e-5)
            a_conv3= tf.nn.leaky_relu(bn_conv3)
            sum_his_W_conv2 = tf.summary.histogram('W_conv3', W_conv3)
            sum_his_b_conv2 = tf.summary.histogram('b_conv3', b_conv3)

            # conv4  #[BS,16,16,256]->[BS,8,8,512]
            W_conv4 = tf.get_variable('W_conv4', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0.))
            z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4
            mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
            offset_conv4= tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
            scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
            bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
            a_conv4= tf.nn.leaky_relu(bn_conv4)
            sum_his_W_conv2 = tf.summary.histogram('W_conv4', W_conv4)
            sum_his_b_conv2 = tf.summary.histogram('b_conv4', b_conv4)

            # flatten  #[BS,8,8,512]->[BS,32768]
            flatten = tf.reshape(a_conv4, [self._BS, 32768])

            # fc1 # classify  #[BS,32768]->[BS,1]
            W_fc1 = tf.get_variable('W_fc1', [flatten.shape[1].value, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('b_fc1', [1], initializer=tf.constant_initializer(0.))
            logits = tf.matmul(flatten, W_fc1) + b_fc1

        self._discriminator_reuse = True
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits

    def build_graph(self):
        # placeholder
        self.random_vec = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.real_image = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        # wgan
        self.fake_image = self._generator(self.random_vec) #[BS,128,128,3]
        real_logits = self._discriminator(self.real_image) #[BS,1]
        fake_logits = self._discriminator(self.fake_image) #[BS,1]
        # loss
        self.d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) #[1]
        self.g_loss = -tf.reduce_mean(fake_logits)  #[1]
        self.sum_his_d_loss = tf.summary.scalar('d_loss', self.d_loss)
        self.sum_his_g_loss = tf.summary.scalar('g_loss', self.g_loss)
        self.sum_merge = tf.summary.merge_all()
        #weight_clip
        self.weight_clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_variables]
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
                    _, _, dLoss, sum_d = sess.run([self.d_optimizer, self.weight_clip_D, self.d_loss, self.sum_his_d_loss],
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


class gp_dc_w_gan(GAN):
    def _generator(self, rand_vec):
        tensor_rand_vec = tf.convert_to_tensor(rand_vec)  #[BS,vec_size]
        with tf.variable_scope('generator'):
            #defc
            W_defc = tf.get_variable('W_defc', [tensor_rand_vec.shape[1].value, 4 * 4 * 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_defc = tf.get_variable('b_defc', [4 * 4 * 1024], initializer=tf.constant_initializer(0.))
            z_defc1 = tf.matmul(tensor_rand_vec, W_defc) + b_defc
            #deflatten  # [BS,4*4*512]->[BS,4,4,512]
            deconv0 = tf.reshape(z_defc1, [-1, 4, 4, 1024])

            mean_conv0, variance_conv0 = tf.nn.moments(deconv0, axes=[0, 1, 2])
            offset_deconv0 = tf.get_variable('offset_deconv0', initializer=tf.zeros([1024]))
            scale_deconv0 = tf.get_variable('scale_deconv0', initializer=tf.ones([1024]))
            bn_deconv0 = tf.nn.batch_normalization(deconv0, mean_conv0, variance_conv0, offset_deconv0, scale_deconv0, 1e-5)
            a_deconv0 = tf.nn.relu(bn_deconv0)

            # deconv1  # [BS,4,4,1024]->[BS,8,8,512]
            W_deconv1 = tf.get_variable('W_deconv1', [5, 5, 512, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv1 = tf.nn.conv2d_transpose(a_deconv0, W_deconv1, [self._BS, 8, 8, 512], [1, 2, 2, 1])
            mean_deconv1, variance_deconv1 = tf.nn.moments(z_deconv1, axes=[0, 1, 2])
            offset_deconv1 = tf.get_variable('offset_deconv1', initializer=tf.zeros([512]))
            scale_deconv1 = tf.get_variable('scale_deconv1', initializer=tf.ones([512]))
            bn_deconv1 = tf.nn.batch_normalization(z_deconv1, mean_deconv1, variance_deconv1, offset_deconv1, scale_deconv1,1e-5)
            a_deconv1 = tf.nn.relu(bn_deconv1)

            # deconv2  # [BS,8,8,512]->[BS,16,16,256]
            W_deconv2 = tf.get_variable('W_deconv2', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv2 = tf.nn.conv2d_transpose(a_deconv1, W_deconv2, [self._BS, 16, 16, 256], [1, 2, 2, 1])
            mean_deconv2, variance_deconv2 = tf.nn.moments(z_deconv2, axes=[0, 1, 2])
            offset_deconv2 = tf.get_variable('offset_deconv2', initializer=tf.zeros([256]))
            scale_deconv2 = tf.get_variable('scale_deconv2', initializer=tf.ones([256]))
            bn_deconv2 = tf.nn.batch_normalization(z_deconv2, mean_deconv2, variance_deconv2, offset_deconv2, scale_deconv2, 1e-5)
            a_deconv2 = tf.nn.relu(bn_deconv2)

            # deconv3  # [BS,16,16,256]->[BS,32,32,128]
            W_deconv3 = tf.get_variable('W_deconv3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv3 = tf.nn.conv2d_transpose(a_deconv2, W_deconv3, [self._BS, 32, 32, 128], [1, 2, 2, 1])
            mean_deconv3, variance_deconv3 = tf.nn.moments(z_deconv3, axes=[0, 1, 2])
            offset_deconv3 = tf.get_variable('offset_deconv3', initializer=tf.zeros([128]))
            scale_deconv3 = tf.get_variable('scale_deconv3', initializer=tf.ones([128]))
            bn_deconv3 = tf.nn.batch_normalization(z_deconv3, mean_deconv3, variance_deconv3, offset_deconv3, scale_deconv3, 1e-5)
            a_deconv3 = tf.nn.relu(bn_deconv3)

            # deconv4  # [BS,32,32,128]->[BS,64,64,64]
            W_deconv4 = tf.get_variable('W_deconv4', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv4 = tf.nn.conv2d_transpose(a_deconv3, W_deconv4, [self._BS, 64, 64, 64], [1, 2, 2, 1])
            mean_deconv4, variance_deconv4 = tf.nn.moments(z_deconv4, axes=[0, 1, 2])
            offset_deconv4 = tf.get_variable('offset_deconv4', initializer=tf.zeros([64]))
            scale_deconv4 = tf.get_variable('scale_deconv4', initializer=tf.ones([64]))
            bn_deconv4 = tf.nn.batch_normalization(z_deconv4, mean_deconv4, variance_deconv4, offset_deconv4, scale_deconv4, 1e-5)
            a_deconv4 = tf.nn.relu(bn_deconv4)

            # deconv5  # [BS,64,64,64]->[BS,128,128,3]
            W_deconv5 = tf.get_variable('W_deconv5', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            z_deconv5 = tf.nn.conv2d_transpose(a_deconv4, W_deconv5, [self._BS, 128, 128, 3], [1, 2, 2, 1])
            a_deconv5 = tf.nn.tanh(z_deconv5)
            self.a_deconv5 = a_deconv5

        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return a_deconv5

    def _discriminator(self, inputs):
        tensor_inputs = tf.convert_to_tensor(inputs)  #[BS,W,H,D]=[BS,128,128,3]
        with tf.name_scope('d'), tf.variable_scope('discriminator', reuse=self._discriminator_reuse):
            # conv1  #[BS,128,128,3]->[BS,64,64,64]
            W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('b_conv1', initializer=tf.constant(0.))
            z_conv1 = tf.nn.conv2d(tensor_inputs / 255, W_conv1, strides=[1, 2, 2, 1],padding='SAME') + b_conv1
            mean_conv1, variance_conv1 = tf.nn.moments(z_conv1, axes=[0, 1, 2])
            offset_conv1 = tf.get_variable('offset_conv1', initializer=tf.zeros([64]))
            scale_conv1 = tf.get_variable('scale_conv1', initializer=tf.ones([64]))
            bn_conv1 = tf.nn.batch_normalization(z_conv1, mean_conv1, variance_conv1,offset_conv1, scale_conv1, 1e-5)
            a_conv1 = tf.nn.leaky_relu(bn_conv1)
            sum_his_W_conv1 = tf.summary.histogram('W_conv1', W_conv1)
            sum_his_b_conv1 = tf.summary.histogram('b_conv1', b_conv1)

            # conv2  #[BS,64,64,64]->[BS,32,32,128]
            W_conv2 = tf.get_variable('W_conv2', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('b_conv2', initializer=tf.constant(0.))
            z_conv2 = tf.nn.conv2d(a_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            mean_conv2, variance_conv2 = tf.nn.moments(z_conv2, axes=[0, 1, 2])
            offset_conv2 = tf.get_variable('offset_conv2',initializer=tf.zeros([128]))
            scale_conv2 = tf.get_variable('scale_conv2', initializer=tf.ones([128]))
            bn_conv2 = tf.nn.batch_normalization(z_conv2, mean_conv2, variance_conv2, offset_conv2, scale_conv2, 1e-5)
            a_conv2 = tf.nn.leaky_relu(bn_conv2)
            sum_his_W_conv2 = tf.summary.histogram('W_conv2', W_conv2)
            sum_his_b_conv2 = tf.summary.histogram('b_conv2', b_conv2)

            # conv3  #[BS,32,32,128]->[BS,16,16,256]
            W_conv3 = tf.get_variable('W_conv3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv3 = tf.get_variable('b_conv3', initializer=tf.constant(0.))
            z_conv3 = tf.nn.conv2d(a_conv2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            mean_conv3, variance_conv3 = tf.nn.moments(z_conv3, axes=[0, 1, 2])
            offset_conv3 = tf.get_variable('offset_conv3', initializer=tf.zeros([256]))
            scale_conv3 = tf.get_variable('scale_conv3', initializer=tf.ones([256]))
            bn_conv3 = tf.nn.batch_normalization(z_conv3, mean_conv3, variance_conv3,offset_conv3, scale_conv3, 1e-5)
            a_conv3= tf.nn.leaky_relu(bn_conv3)
            sum_his_W_conv2 = tf.summary.histogram('W_conv3', W_conv3)
            sum_his_b_conv2 = tf.summary.histogram('b_conv3', b_conv3)

            # conv4  #[BS,16,16,256]->[BS,8,8,512]
            W_conv4 = tf.get_variable('W_conv4', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv4 = tf.get_variable('b_conv4', initializer=tf.constant(0.))
            z_conv4 = tf.nn.conv2d(a_conv3, W_conv4, strides=[1, 2, 2, 1], padding='SAME') + b_conv4
            mean_conv4, variance_conv4 = tf.nn.moments(z_conv4, axes=[0, 1, 2])
            offset_conv4= tf.get_variable('offset_conv4', initializer=tf.zeros([512]))
            scale_conv4 = tf.get_variable('scale_conv4', initializer=tf.ones([512]))
            bn_conv4 = tf.nn.batch_normalization(z_conv4, mean_conv4, variance_conv4, offset_conv4, scale_conv4, 1e-5)
            a_conv4= tf.nn.leaky_relu(bn_conv4)
            sum_his_W_conv2 = tf.summary.histogram('W_conv4', W_conv4)
            sum_his_b_conv2 = tf.summary.histogram('b_conv4', b_conv4)

            # flatten  #[BS,8,8,512]->[BS,32768]
            flatten = tf.reshape(a_conv4, [self._BS, 32768])

            # fc1 # classify  #[BS,32768]->[BS,1]
            W_fc1 = tf.get_variable('W_fc1', [flatten.shape[1].value, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('b_fc1', [1], initializer=tf.constant_initializer(0.))
            logits = tf.matmul(flatten, W_fc1) + b_fc1

        self._discriminator_reuse = True
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits

    def build_graph(self):
        # placeholder
        self.random_vec = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.real_image = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        # wgan
        self.fake_image = self._generator(self.random_vec) #[BS,128,128,3]
        real_logits = self._discriminator(self.real_image) #[BS,1]
        fake_logits = self._discriminator(self.fake_image) #[BS,1]
        # d_gradient_penalty
        alpha = tf.random_uniform(shape=[self._BS, 1, 1, 1], minval=0., maxval=1.)
        interpolates = tf.multiply(alpha, self.real_image) + tf.multiply((1 - alpha), self.fake_image) #[BS,128,128,3]
        interpolates_logits = self._discriminator(interpolates) #[BS,1]
        gradients = tf.gradients(interpolates_logits, [interpolates])[0] #[BS,128,128,3]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2,3])) #[BS,1]
        d_gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
        # loss
        self.d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + 0.1*d_gradient_penalty #[1]
        self.g_loss = -tf.reduce_mean(fake_logits)  #[1]
        self.sum_his_d_loss = tf.summary.scalar('d_loss', self.d_loss)
        self.sum_his_g_loss = tf.summary.scalar('g_loss', self.g_loss)
        self.sum_merge = tf.summary.merge_all()
        # optimize
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_variables)
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_variables)


if __name__ == "__main__":
    gan = gp_dc_w_gan(32, 100)
    gan.build_graph()
    gan.train()
