import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_net import NET
from tensorflow_net_test import NET as NET_test
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


net = NET_test()
tf_sum_writer = tf.summary.FileWriter('logs/without_BN')
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir='tfModel_0/')


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config = tf_config) as sess:
    tf_sum_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if ckpt and ckpt.model_checkpoint_path:
        print('loading_model')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('no_pre_model')

    for train_step in range(int(1e6)):
        batch_input, batch_target = mnist.train.next_batch(32)
        _,net_predict,net_train_sum=sess.run([net.optimize,net.predict,net.sum_merge_train],
            feed_dict={net.flatten_im:batch_input,net.target:batch_target,net.keep_prob:0.5})
        tf_sum_writer.add_summary(net_train_sum, train_step)

        test_input = mnist.test.images[:1000]
        test_target = mnist.test.labels[:1000]
        net_accuracy,net_test_sum = sess.run([net.accuracy, net.sum_merge_test],
            feed_dict={net.flatten_im: test_input, net.target: test_target, net.keep_prob: 1})
        tf_sum_writer.add_summary(net_test_sum, train_step)

        print(train_step, net_accuracy)

        if train_step % 1000 == 0:  # 10k
            saver.save(sess, 'tfModel_0/model.ckpt', global_step=train_step)
