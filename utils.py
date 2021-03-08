import cv2
import numpy
import sys, os, time
import random

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

def get_image_paths( directory ):
    return [ x.path for x in os.scandir( directory ) if x.name.endswith(".jpg") or x.name.endswith(".png") ]

def get_transpose_axes( n ):
    if n % 2 == 0:
        y_axes = list( range( 1, n-1, 2 ) )
        x_axes = list( range( 0, n-1, 2 ) )
    else:
        y_axes = list( range( 0, n-1, 2 ) )
        x_axes = list( range( 1, n-1, 2 ) )
    return y_axes, x_axes, [n-1]

def stack_images( images ):
    images_shape = numpy.array( images.shape )
    new_axes = get_transpose_axes( len( images_shape ) )
    new_shape = [ numpy.prod( images_shape[x] ) for x in new_axes ]
    return numpy.transpose(
        images,
        axes = numpy.concatenate( new_axes )
        ).reshape( new_shape )
    

test_batch_percentage = 0.1
img_size = 256
channels = 3
batch_size = 64#16
# TIMING CONTROL
def print_time(start_time):
    elapsed_time = time.time() - start_time
    mins = int(elapsed_time / 60)
    secs = elapsed_time - (mins * 60)
    print("Accumulative time: %02d:%02d" % (mins, int(secs % 60)))
    
#################################    TF IMAGE PREPROCESS   ####################################

def get_tf_images(path,is_norm=False):
    images = get_image_paths( path )

    # create a partition vector
    partitions = [0] * len(images)
    partitions[:int(len(images) * test_batch_percentage)] = [1] * int(len(images) * test_batch_percentage)
    random.shuffle(partitions)

    # convert string into tensors
    all_images = ops.convert_to_tensor(images, dtype=dtypes.string)

    # partition our data into a test and train set according to our partition vector
    train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)

    # create input queues
    train_input_queue = tf.train.slice_input_producer([train_images], shuffle=True)
    test_input_queue = tf.train.slice_input_producer([test_images], shuffle=True)

    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=channels)
#    train_image = tf.image.resize(train_image, [img_size, img_size]) #Add
    
    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content, channels=channels)
#    test_image = tf.image.resize(test_image, [img_size, img_size]) #Add

    # define tensor shape
    train_image.set_shape([img_size, img_size, channels])
    test_image.set_shape([img_size, img_size, channels])

    # collect batches of images before processing
    train_image_batch = tf.train.batch([train_image], batch_size=batch_size)
    test_image_batch = tf.train.batch([test_image], batch_size=int(len(images) * test_batch_percentage))
    
    if is_norm:
        train_image_batch = tf.cast(train_image_batch, tf.float32) / 255.0 #Add
        test_image_batch = tf.cast(test_image_batch, tf.float32) / 255.0  #Add
    
    return train_image_batch, test_image_batch

