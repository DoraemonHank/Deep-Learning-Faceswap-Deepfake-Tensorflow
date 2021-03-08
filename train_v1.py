import cv2
import tensorflow as tf
import numpy as np
import os
from utils import stack_images,print_time,get_tf_images
from models import Autoencoder_v1
import matplotlib.pyplot as plt
from training_data import get_training_data


saved_ckpt_path = './checkpoint/'
img_size = 64
channels = 3
batch_size = 64
training_epochs = 2000000

with tf.device("/gpu:0"):

    # TF IMAGE PREPROCESS
    train_image_batch_A, test_image_batch_A = get_tf_images("data/trump/",True)
    train_image_batch_B, test_image_batch_B = get_tf_images("data/cage/",True)
    train_image_batch_A += tf.reduce_mean( train_image_batch_B ) -  tf.reduce_mean( train_image_batch_A )
    
    # INPUTS AND DESIRED OUTPUT
    X = tf.placeholder(tf.float32, [None, img_size, img_size, channels],name="input")
    Y = tf.placeholder(tf.float32, [None, img_size, img_size, channels], name="prediction")
    
    
    autoencoder = Autoencoder_v1()
    enc = autoencoder.Encoder(X,'encoder',True)
    pred_A = Autoencoder_v1().Decoder(enc,'decoder_A', False)
    pred_B = Autoencoder_v1().Decoder(enc,'decoder_B', False)
    
    loss_all_A,optimizer_A = Autoencoder_v1().forward(pred_A,Y,'loss_A')
    loss_all_B,optimizer_B = Autoencoder_v1().forward(pred_B,Y,'loss_B')

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
            warped_A, target_A = get_training_data( batch_images, batch_size )


            feeds = {X: warped_A, Y: target_A}
            sess.run(optimizer_A, feed_dict=feeds)

            # TRAIN ENCONDER + DECODER_B WITH BATCH_B
            batch_images = sess.run(train_image_batch_B)
            warped_B, target_B = get_training_data( batch_images, batch_size )

            feeds = {X: warped_B, Y: target_B}
            sess.run(optimizer_B, feed_dict=feeds)

            print("Epoch %02d/%02d average cost: %.4f"
                   % (epoch, training_epochs, sess.run(loss_all_B, \
                                                       feed_dict={X: warped_B,\
                                                                  Y: target_B})))
            
            if epoch % 100 == 0:
                saver_encoder.save(sess, os.path.join(saved_ckpt_path + 'encoder/', str(epoch) + 'saved_encoder'), global_step=epoch)
                saver_decoder_A.save(sess, os.path.join(saved_ckpt_path + 'decoder_A/', str(epoch) + 'saved_decoder_A'), global_step=epoch)
                saver_decoder_B.save(sess, os.path.join(saved_ckpt_path + 'decoder_B/', str(epoch) + 'saved_decoder_B'), global_step=epoch)
                
                test_A = sess.run(test_image_batch_A)
                test_B = sess.run(test_image_batch_B)
                
                _,test_A = get_training_data(test_A,(test_A.shape)[0])
                _,test_B = get_training_data(test_B,(test_B.shape)[0])

            predAA = sess.run(pred_A, feed_dict={X: test_A})
            predAB = sess.run(pred_B, feed_dict={X: test_A})
            
            predBA = sess.run(pred_A, feed_dict={X: test_B})
            predBB = sess.run(pred_B, feed_dict={X: test_B})

            figure_A = np.stack([
                test_A[0:14],
                predAA[0:14],
                predAB[0:14],
                ], axis=1 )
            figure_B = np.stack([
                test_B[0:14],
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


