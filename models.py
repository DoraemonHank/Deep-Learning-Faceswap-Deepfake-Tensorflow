import tensorflow as tf
import numpy as np

ENCODER_DIM = 1024

class Layer():

    def __init__(self, shape, stddev, value):
        self.weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
        self.biases = tf.Variable(tf.constant(value=value, shape=[shape[-1]],dtype=tf.float32))

    def feed_forward(self, input_data, stride=None):
        raise NotImplementedError

class Convolution_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Convolution_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding="SAME")
        output_data = tf.nn.leaky_relu(conv, alpha=0.1)
        return output_data

def _ConvLayer(input_size, output_size):
    def block(x):
        low_level_conv = Convolution_Layer(shape=[5, 5, input_size, output_size], stddev=0.01, value=0.1)
        output = low_level_conv.feed_forward(input_data=x,stride = [1,2,2,1])
        return output
    return block

def _Upscale( input_size,output_features ):
    def block(x):
        weights = tf.Variable(tf.truncated_normal(shape=[3,3,input_size,output_features*4], stddev=0.01,dtype=tf.float32))
        upscale = tf.nn.conv2d(x, weights, padding="SAME")
        x = tf.nn.leaky_relu(upscale, alpha=0.1)
        x = tf.depth_to_space(x,2)
        return x
    return block

class Autoencoder_v1():

    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def Encoder(self,input,name, reuse=True):
        with tf.device("/gpu:0"):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = _ConvLayer(3,128)(input)
                x = _ConvLayer(128,256)(x)
                x = _ConvLayer(256,512)(x)
                x = _ConvLayer(512,1024)(x)
                x = tf.layers.flatten(x)
                
                # fully connected layer 1
                W_fc1 = self.weight_variable([16384, ENCODER_DIM])
                x = tf.matmul(x, W_fc1)
                
                # fully connected layer 2
                W_fc2 = self.weight_variable([ENCODER_DIM, 4*4*1024])
                x = tf.matmul(x, W_fc2)
                
                x = tf.reshape(x, [-1,4,4,1024])
                x = _Upscale(1024,512)(x)
                return x

    def Decoder(self,input,name, reuse=False):
        with tf.device("/gpu:0"):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                x = _Upscale(512,256)(input)
                x = _Upscale(256,128)(x)
                x = _Upscale(128,64)(x)
                weights = tf.Variable(tf.truncated_normal(shape=[5,5,64,3], stddev=0.01))
                x = tf.nn.conv2d(x,weights,padding="SAME")
                x = tf.nn.sigmoid(x,name="sigmoid")
                return x
    
    def forward(self, logits,ground_truth,name):
        with tf.device("/gpu:0"):
            with tf.variable_scope(name):
                loss_all = tf.reduce_mean(tf.abs(tf.subtract(logits, ground_truth)))  ## MEAN ABSOLUTE ERROR
                optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999).minimize(loss_all)
                return loss_all,optimizer
            
class Autoencoder_v2():
    def __init__(self):
        self.n1 = 128
        self.n2 = 256
        self.n3 = 512
        self.channels = 3
        self.learning_rate = 0.001
    def encoder(self,_X, _keepprob, name, reuse=True):
        with tf.device("/gpu:0"):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                    
                _W = {
                    'ce1': tf.Variable(tf.random_normal([4, 4, self.channels, self.n1], stddev=0.1)),
                    'ce2': tf.Variable(tf.random_normal([4, 4, self.n1, self.n2], stddev=0.1)),
                    'ce3': tf.Variable(tf.random_normal([4, 4, self.n2, self.n3], stddev=0.1))
                }
                
                _b = {
                    'be1': tf.Variable(tf.random_normal([self.n1], stddev=0.1)),
                    'be2': tf.Variable(tf.random_normal([self.n2], stddev=0.1)),
                    'be3': tf.Variable(tf.random_normal([self.n3], stddev=0.1))
                }
    
                _ce1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(_X, _W['ce1'], strides=[1,4,4,1], padding='SAME'), _b['be1']))
                _ce1 = tf.nn.dropout(_ce1, _keepprob)
                _ce2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(_ce1, _W['ce2'], strides=[1,4,4,1], padding='SAME'), _b['be2']))
                _ce2 = tf.nn.dropout(_ce2, _keepprob)
                _ce3 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(_ce2, _W['ce3'], strides=[1,4,4,1], padding='SAME'), _b['be3']))
                _ce3 = tf.nn.dropout(_ce3, _keepprob)
        return _ce3
    
    def decoder(self,encoder_input, _X, _keepprob, name, reuse=False):
        with tf.device("/gpu:0"):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()
                    
                _W = {
                    'cd3': tf.Variable(tf.random_normal([4, 4, self.n2, self.n3], stddev=0.1)),
                    'cd2': tf.Variable(tf.random_normal([4, 4, self.n1, self.n2], stddev=0.1)),
                    'cd1': tf.Variable(tf.random_normal([4, 4, self.channels, self.n1], stddev=0.1))
                }
                
                _b = {
                    'bd3': tf.Variable(tf.random_normal([self.n2], stddev=0.1)),
                    'bd2': tf.Variable(tf.random_normal([self.n1], stddev=0.1)),
                    'bd1': tf.Variable(tf.random_normal([self.channels], stddev=0.1))
                }
    
                _cd3 = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(encoder_input, _W['cd3'],\
                                                                   tf.stack([_X, 16, 16, self.n2]),\
                                                                   strides=[1,4,4,1], padding='SAME'), _b['bd3']))
                _cd3 = tf.nn.dropout(_cd3, _keepprob)
                _cd2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(_cd3, _W['cd2'],\
                                                                   tf.stack([_X, 64, 64, self.n1]),\
                                                                   strides=[1,4,4,1], padding='SAME'), _b['bd2']))
                _cd2 = tf.nn.dropout(_cd2, _keepprob)
                _cd1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(_cd2, _W['cd1'],\
                                                                   tf.stack([_X, 256, 256, 3]),\
                                                                   strides=[1,4,4,1], padding='SAME'), _b['bd1']))
                _cd1 = tf.nn.dropout(_cd1, _keepprob)
                _out = _cd1
        return _out
    
    ############################    COST FUNCTION AND OPTIMIZER   #################################
    
    def cost_func(self,pred, Y,name):
        with tf.device("/gpu:0"):
            # Prediction have a sigmoid so it's limited to 0-1 and usual image are between 0-255 each pixel
            cost = tf.reduce_mean(tf.abs(tf.subtract(pred, tf.divide(Y, 255))))  ## MEAN ABSOLUTE ERROR
    #         cost = tf.losses.mean_squared_error(tf.divide(Y, 255), pred) ## MEAN SQUARRED ERROR
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
    
            # SUMMARIES
            tf.summary.scalar(name, cost)
            summary_op = tf.summary.merge_all()
            return cost, optimizer
    
if __name__ == '__main__':
    
    x = tf.constant(1.0, shape=[6, 64, 64, 3])
    y = tf.constant(1.0, shape=[6, 64, 64, 3])

    autoencoder = Autoencoder_v1()
    logits = autoencoder.forward(x,y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2):
            y_val = sess.run(logits,feed_dict={})
            print(y_val)




