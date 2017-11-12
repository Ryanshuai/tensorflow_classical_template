import tensorflow as tf
import os
import numpy as np
import cv2

current_dir = os.getcwd()
# parent = os.path.dirname(current_dir)
pokemon_dir = os.path.join(current_dir, 'data')
image_paths = []
for each in os.listdir(pokemon_dir):
    image_paths.append(os.path.join(pokemon_dir, each))
tensor_image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def preprocessing(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string)
    image = tf.image.resize_images(image, [128, 128])
    image.set_shape([128, 128, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image



dataset = tf.data.Dataset.from_tensor_slices(tensor_image_paths)
dataset = dataset.repeat(10)
dataset = dataset.map(preprocessing)
dataset = dataset.shuffle(buffer_size=10000)
#dataset = dataset.batch(32)

iterator = dataset.make_initializable_iterator()

print('**************************')
print(iterator.output_shapes)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    global_step = 0
    for epoch in range(5000):
        sess.run(iterator.initializer)
        print('**************************')
        print(iterator.output_shapes)
        epoch_step = 0
        while True:
            try:
                real_image = sess.run(iterator.get_next())
                print(real_image)
                print('**************************')
                print(iterator.output_shapes)
                print('---------------------')
                print(real_image.shape)
            except tf.errors.OutOfRangeError:
                break

            print('epoch:', epoch, 'epoch_step:', epoch_step, 'global_step:', global_step)