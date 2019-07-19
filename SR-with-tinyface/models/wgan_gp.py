# coding: utf-8
import tensorflow as tf
import numpy as np
import scipy.misc
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)

+ GP:
Instead of weight clipping, WGAN-GP proposed gradient penalty.
'''

class WGAN_GP(BaseModel):
    def __init__(self, name, training,D_lr=2*1e-4, G_lr=2*1e-4, image_shape=[128, 128, 3], z_dim=[128,128,3]):
        self.beta1 = 0.5
        self.beta2 = 0.9
        #self.D_lr=2*1e-4
        #self.G_lr=2*1e-4
        self.ld = 10. # lambda
        self.n_critic = 1
        super(WGAN_GP, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            # change by xjc z_dim 64 -> [8,8,3]
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None] + self.z_dim)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # `critic` named from wgan (wgan-gp use the term `discriminator` rather than `critic`)
            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)

            W_dist = tf.reduce_mean(C_real - C_fake)


            # fea_X_norm = tf.divide(fea_X, tf.norm(fea_X, ord = 'euclidean'))
            # fea_G_norm = tf.divide(fea_G, tf.norm(fea_G, ord = 'euclidean'))
            # L2_dist = tf.sqrt(tf.reduce_sum(tf.square(fea_X - fea_G)))   #(batch,1)
            
            # tf.norm(slim.flatten(C_xhat_grad), axis=1)p
            
            gen_l1_cost = tf.reduce_mean(tf.abs(G - X))
            #gen_l2_cost = tf.reduce_mean(tf.square(G - X))
            gen_l2_cost = tf.sqrt(tf.reduce_sum(tf.square(G - X)))
            #gen_l2_cost = tf.reduce_sum(tf.square(G - X))
            
            #M = scipy.misc.imresize(G,[16,16,3])
            
            
            #gen_l2_cost1 = tf.sqrt(tf.reduce_sum(tf.square(M - Z)))
            
            C_loss = -W_dist
            #- L2_dist
            
            #G_loss = 0.2 * tf.reduce_mean(-C_fake) + 0.8 * gen_l2_cost #比例可改
            G_loss = 0.4 * tf.reduce_mean(-C_fake) + 0.6* gen_l2_cost
            #G_loss = tf.reduce_mean(-C_fake)
            # add by xjc MSE_loss
            # MSE_loss = tf.reduce_mean(slim.losses.mean_squared_error(predictions=G, labels=X, weights=1.0)) 
            # G_loss += MSE_loss
            # Gradient Penalty (GP)
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = eps*X + (1.-eps)*G 
            C_xhat = self._critic(x_hat, reuse=True)
            C_xhat_grad = tf.gradients(C_xhat, x_hat)[0] # gradient of D(x_hat)
            C_xhat_grad_norm = tf.norm(slim.flatten(C_xhat_grad), axis=1)  # l2 norm
            #GP = self.ld * tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.))
            #C_loss += GP
            zzz = tf.zeros(tf.shape(C_xhat_grad_norm), dtype=tf.float32,name=None)
            LP = self.ld * tf.reduce_mean(tf.square(tf.maximum(zzz,(C_xhat_grad_norm - 1.))))
            C_loss += LP

            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            n_critic = 1
            lr = 2*1e-4
            learning_rate = tf.train.exponential_decay(lr, global_step,5000, 0.9, staircase=True)
            #learning_rate = 0.2*1e-4
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=self.beta1, beta2=self.beta2).\
                    minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=self.beta1, beta2=self.beta2).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                #tf.summary.scalar(' L1_dist', gen_l1_cost),
                tf.summary.scalar(' L2_dist', gen_l2_cost),
                tf.summary.scalar(' G_adversial', tf.reduce_mean(-C_fake)),
                # tf.summary.scalar('MSE_loss', MSE_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('C_real', tf.reduce_mean(C_real)),
                tf.summary.scalar('C_fake', tf.reduce_mean(C_fake)),
                tf.summary.scalar('W_dist', W_dist),
                tf.summary.scalar('LP', LP),
                tf.summary.scalar('lr',learning_rate)
                #tf.summary.scalar('G_lr',G_lr)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('lr_sample', z, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('hr_sample', X, max_outputs=self.FAKE_MAX_OUTPUT)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = C_train_op # train.py 와의 accesibility 를 위해... 흠... 구린데...
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        #return self._good_critic(X, reuse)
        return self._good_critic(X,reuse)
    def _generator(self, z, reuse=False):
        return self._good_generator(z, reuse)
        # return self._good_generator(X,reuse)
    def _dcgan_critic(self, X, reuse=False):
        '''
        K-Lipschitz function.
        WGAN-GP does not use critic in batch norm.
        '''
        with tf.variable_scope('critic', reuse=reuse):
            net = X
            
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, padding='SAME', activation_fn=ops.lrelu):
                net = slim.conv2d(net, 64)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])

            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)

            return net

    def _dcgan_generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=2, activation_fn=tf.nn.relu, 
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])

                return net

    '''
    ResNet architecture from appendix C in the paper.
    https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py - GoodGenerator / GoodDiscriminator
    layer norm in D, batch norm in G.
    some details are ignored in this implemenation.
    '''
 
    # xian juanji zai ps
    def _residual_block(self, X, nf_output, resample, kernel_size=[3,3], name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
            nf_input = input_shape[-1]

            if resample == 'down': # Downsample
                shortcut = slim.avg_pool2d(X, [2,2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) # init xavier

                #net = slim.layer_norm(X, activation_fn=tf.nn.relu)
                net = slim.layer_norm(X, activation_fn=ops.lrelu)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None,activation_fn=None) # skip bias
                #net = slim.layer_norm(net, activation_fn=tf.nn.relu)
                net = slim.layer_norm(net, activation_fn=ops.lrelu)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size,activation_fn=None)
                net = slim.avg_pool2d(net, [2,2])
             
                return net + shortcut
            if resample == 'down1': # Downsample
                shortcut = slim.avg_pool2d(X, [2,2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) # init xavier

                #net = slim.layer_norm(X, activation_fn=tf.nn.relu)
                #net = slim.batch_norm(X, activation_fn=ops.lrelu, **self.bn_params)
                net = slim.layer_norm(X, activation_fn=ops.lrelu)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None,activation_fn=None) # skip bias
                #net = slim.layer_norm(net, activation_fn=tf.nn.relu)
                #net = slim.batch_norm(X, activation_fn=ops.lrelu, **self.bn_params)
                net = slim.layer_norm(net, activation_fn=ops.lrelu)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size,activation_fn=None)
                net = slim.avg_pool2d(net, [2,2])

                return net + shortcut
            elif resample == 'up': # Upsample
                upsample_shape = [int(x)*2 for x in input_shape[1:3]]
                shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape) #dui x chaongfu 
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) #ke xuan xiang
                

                #net = slim.batch_norm(X, activation_fn=ops.lrelu, **self.bn_params) 
                net = slim.layer_norm(X, activation_fn=ops.lrelu)
                net = tf.image.resize_nearest_neighbor(net, upsample_shape)  
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None) # skip bias 
                #net = slim.batch_norm(net, activation_fn=ops.lrelu, **self.bn_params) 
                net = slim.layer_norm(net, activation_fn=ops.lrelu)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size) 


                return net + shortcut
            elif resample == "same":
                net = slim.batch_norm(X, activation_fn=tf.nn.relu, **self.bn_params)
                net = slim.conv2d(X, nf_output, kernel_size=kernel_size, biases_initializer=None,activation_fn=ops.lrelu) # skip bias
                net = slim.batch_norm(_net, activation_fn=tf.nn.relu, **self.bn_params)
                net = slim.conv2d(_net, nf_output, kernel_size=kernel_size,activation_fn=ops.lrelu)
                return X + net
            else:
                raise Exception('invalid resample value')
    
   


    
    def _good_critic(self, X, reuse=False):
        with tf.variable_scope('critic', reuse=reuse):
            nf = 64
            #net = slim.conv2d(X, 1*nf, [3,3], padding='SAME',activation_fn=ops.lrelu)
            #net = slim.avg_pool2d(net, [2,2])#64
            #net = slim.conv2d(net, 2*nf, [3,3], padding='SAME',activation_fn=ops.lrelu)
            #net = slim.avg_pool2d(net, [2,2])#32
            #net = slim.conv2d(net, 2*nf, [3,3], padding='SAME',activation_fn=ops.lrelu)
            #net = slim.avg_pool2d(net, [2,2])#16
            #net = slim.conv2d(net, 4*nf, [3,3], padding='SAME',activation_fn=ops.lrelu)
            #net = slim.avg_pool2d(net, [2,2])#8
            #net = slim.conv2d(net, 8*nf, [3,3], padding='SAME',activation_fn=ops.lrelu)
            #net = slim.conv2d(net, 1, [1,1], activation_fn=None)
            #net = slim.avg_pool2d(net, [4,4])#1
            net = slim.conv2d(X, nf, [3,3], padding='SAME',activation_fn=ops.lrelu) # 
            net = self._residual_block(net, 1*nf, resample='down', name='res_block1') # 64x64x64
            #net = slim.dropout(net,0.2)
            net = self._residual_block(net, 2*nf, resample='down', name='res_block2') # 32
            #net = slim.dropout(net,0.2)
            net = self._residual_block(net, 2*nf, resample='down', name='res_block3') # 16
            #net = slim.dropout(net,0.2)
            net = self._residual_block(net, 4*nf, resample='down', name='res_block4') # 8
            #net = slim.dropout(net,0.2)
            net = self._residual_block(net, 8*nf, resample='down', name='res_block5') # 4
            #net = slim.dropout(net,0.2)
            #net = self._residual_block(net, 4*nf, resample='down', name='res_block6')
            #net = self._residual_block(net, 4*nf, resample='down', name='res_block7')
            net = slim.flatten(net)
            net = slim.fully_connected(net,1024,activation_fn=ops.lrelu)
            net = slim.fully_connected(net,512,activation_fn=ops.lrelu)
            #net = slim.dropout(net)
            net = slim.fully_connected(net,1,activation_fn=None)
        
            return net


    def _good_generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            nf = 64
            net = z
            net = self._residual_block(net, 1*nf, resample='down1', name='res_block1') # 64x64 
            net1 = net
            net = self._residual_block(net, 2*nf, resample='down1', name='res_block2') # 32x32
            net2 = net
            net = self._residual_block(net, 3*nf, resample='down1', name='res_block3') # 16x16
            net3 = net
            net = self._residual_block(net, 4*nf, resample='down1', name='res_block4') # 8x8
            net4 = net

            net = slim.conv2d(net, 768, kernel_size=[8,8],padding='valid', activation_fn=ops.lrelu)
            print(net.shape)
            net = slim.conv2d(net, 8*8*nf, kernel_size=[1,1], padding='valid',activation_fn=ops.lrelu)
            print(net.shape)
            net  = tf.reshape(net, [ -1,8, 8, nf])

            
            #net2 = tf.image.resize_nearest_neighbor(net2, [128,128])
            net = tf.concat([net, net4], 3)
            net = self._residual_block(net, 4*nf, resample='up', name='res_blocks1') # 16x16
            
            net = tf.concat([net, net3], 3) 
            net = self._residual_block(net, 3*nf, resample='up', name='res_blocks2') # 32x32
            
            net = tf.concat([net, net2], 3)
            net = self._residual_block(net, 2*nf, resample='up', name='res_blocks3') # 64x64

            net = tf.concat([net, net1], 3)
            net = self._residual_block(net, 1*nf, resample='up', name='res_blocks4') # 128x128
            #net = slim.conv2d(net,nf ,[1,1],padding='SAME', activation_fn=ops.lrelu)
            
            #net = tf.concat([net, net1, net2, net3], 3) 
             
            print(net.shape)
            #expected_shape(net, [64, 64, 64])
            #net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
            net = slim.conv2d(net, 3, kernel_size=[3,3], activation_fn=tf.nn.tanh)
            #expected_shape(net, [64, 64, 3])
            print(net.shape)
            return net

    
    
    
