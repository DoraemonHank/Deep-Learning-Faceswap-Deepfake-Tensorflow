import cv2
import tensorflow as tf
import numpy as np
import os
from utils import stack_images,print_time,get_tf_images
from models import Autoencoder_v2
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES']='0'

saved_ckpt_path = './MODEL/'
img_size = 256
channels = 3
training_epochs = 2000000

with tf.device("/gpu:0"):

    # TF IMAGE PREPROCESS
    train_image_batch_A, test_image_batch_A = get_tf_images("data/trump/")
    train_image_batch_B, test_image_batch_B = get_tf_images("data/cage/")
    
    # INPUTS AND DESIRED OUTPUT
    X = tf.placeholder(tf.float32, [None, img_size, img_size, channels],name="input")
    Y = tf.placeholder(tf.float32, [None, img_size, img_size, channels], name="prediction")
    dropout_keep_prob = tf.placeholder(tf.float32)
    
    autoencoder = Autoencoder_v2()
    enc = autoencoder.encoder (X, dropout_keep_prob, 'encoder', True)
    pred_A = autoencoder.decoder(enc, tf.shape(X)[0], dropout_keep_prob, 'decoder_A', False)
    pred_B = autoencoder.decoder(enc, tf.shape(X)[0], dropout_keep_prob, 'decoder_B', False)
    
    # COST & OPTIMIZERS
    cost_A, optimizer_A = autoencoder.cost_func(pred_A, Y,"cost_A")
    cost_B, optimizer_B = autoencoder.cost_func(pred_B, Y,"cost_B")
    
    
    start_step = 0
    
    
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
    with tf.Session(config = config) as sess:
        
    #    sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        t_vars = tf.trainable_variables()
        encoder_vars = [var for var in t_vars if 'encoder' in var.name]
        decoder_A_vars = [var for var in t_vars if 'decoder_A' in var.name]
        decoder_B_vars = [var for var in t_vars if 'decoder_B' in var.name]
        
        saver_encoder = tf.train.Saver(encoder_vars)
        saver_decoder_A = tf.train.Saver(decoder_A_vars)
        saver_decoder_B = tf.train.Saver(decoder_B_vars)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        #saver = tf.train.Saver()
        if os.path.exists(saved_ckpt_path + 'encoder/'):
            ckpt = tf.train.get_checkpoint_state(saved_ckpt_path + 'encoder/')
            if ckpt and ckpt.model_checkpoint_path:
                saver_encoder.restore(sess, ckpt.model_checkpoint_path)
                start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print("Model encoder restored...")
        
        if os.path.exists(saved_ckpt_path + 'decoder_A/'):
            ckpt = tf.train.get_checkpoint_state(saved_ckpt_path + 'decoder_A/')
            if ckpt and ckpt.model_checkpoint_path:
                saver_decoder_A.restore(sess, ckpt.model_checkpoint_path)
                print("Model decoder_A restored...")
                
        if os.path.exists(saved_ckpt_path + 'decoder_B/'):
            ckpt = tf.train.get_checkpoint_state(saved_ckpt_path + 'decoder_B/')
            if ckpt and ckpt.model_checkpoint_path:
                saver_decoder_B.restore(sess, ckpt.model_checkpoint_path)
                print("Model saver_decoder_B restored...")
    #    saver = tf.train.import_meta_graph(saved_ckpt_path + '66700model-66700.meta')
    #    saver.restore(sess,tf.train.latest_checkpoint(saved_ckpt_path))
    
    #    train_summaryA_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    #    train_summaryB_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    
        print( "press 'q' to stop training and save model" )
        
        for epoch in range(start_step,training_epochs):
            # TRAIN ENCONDER + DECODER_A WITH BATCH_A
            batch_images = sess.run(train_image_batch_A)
            feeds = {X: batch_images, Y: batch_images, dropout_keep_prob: 0.7}
            sess.run(optimizer_A, feed_dict=feeds)
        
            # TRAIN ENCONDER + DECODER_B WITH BATCH_B
            batch_images = sess.run(train_image_batch_B)
            feeds = {X: batch_images, Y: batch_images, dropout_keep_prob: 0.7}
            sess.run(optimizer_B, feed_dict=feeds)
    
    
            print("Epoch %02d/%02d average cost: %.4f"
                   % (epoch, training_epochs, sess.run(cost_B, \
                                                       feed_dict={X: batch_images,\
                                                                  Y: batch_images, dropout_keep_prob: 1})))
            
            if epoch % 100 == 0:
    #            save_model_weights()
                #saver.save(sess, os.path.join(saved_ckpt_path, str(epoch) + 'model'), global_step=epoch)
                saver_encoder.save(sess, os.path.join(saved_ckpt_path + 'encoder/', str(epoch) + 'saved_encoder'), global_step=epoch)
                saver_decoder_A.save(sess, os.path.join(saved_ckpt_path + 'decoder_A/', str(epoch) + 'saved_decoder_A'), global_step=epoch)
                saver_decoder_B.save(sess, os.path.join(saved_ckpt_path + 'decoder_B/', str(epoch) + 'saved_decoder_B'), global_step=epoch)
                
                test_A = sess.run(test_image_batch_A)
                test_B = sess.run(test_image_batch_B)
                
                predAA = sess.run(pred_A, feed_dict={X: test_A, dropout_keep_prob: 1})
                predAB = sess.run(pred_B, feed_dict={X: test_A, dropout_keep_prob: 1})
                
                predBA = sess.run(pred_A, feed_dict={X: test_B, dropout_keep_prob: 1})
                predBB = sess.run(pred_B, feed_dict={X: test_B, dropout_keep_prob: 1})
                
#                fig, axs = plt.subplots(2, 4, figsize=(30, 16))
#                for example_i in range(4):
#                    axs[0, example_i].imshow(test_A[example_i])
#                    axs[1, example_i].imshow(predAB[example_i])
#    
#                plt.show()
#                test_A = np.clip( test_A * 1, 0, 255 ).astype('uint8')
#                test_B = np.clip( test_B * 1, 0, 255 ).astype('uint8')
            figure_A = np.stack([
                test_A[0:14]/255,
                predAA[0:14],
                predAB[0:14],
                ], axis=1 )
            figure_B = np.stack([
                test_B[0:14]/255,
                predBB[0:14],
                predBA[0:14],
                ], axis=1 )
        
            figure = np.concatenate( [ figure_A, figure_B ], axis=0 )
            figure = figure.reshape( (4,7) + figure.shape[1:] )
            figure = stack_images( figure )
        
            figure = np.clip( figure * 255, 0, 255 ).astype('uint8')
            
            figure = figure[:, :, ::-1].copy() 
            cv2.namedWindow("frame",0)
            cv2.imshow( "frame", figure )
            key = cv2.waitKey(1)
            if key == ord('q'):
    #            save_model_weights()
                exit()


