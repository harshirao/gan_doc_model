import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

############
def harmonic_mean(n,t):
    res=0
    for i in range(n):
        res=res+float(1)/t[i]
        res=n/res
    return res

############
def vectors(model, data, session):
    vecs = []
    for _, x in data:
        vecs.extend(
            session.run([model.rep], feed_dict={
                model.x: x
            })[0]
        )
    return np.array(vecs)

############
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

############
def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

############
def generator(z, size, output_size, ith_gen):
    h1 = tf.nn.leaky_relu(linear(z, size, ("g_%d_0" %ith_gen)), alpha=0.2)           # first layer 
    G_batchnorm = tf.layers.batch_normalization(h1, training=False)                   
    h2 = tf.nn.leaky_relu(linear(G_batchnorm, size, ("g_%d_1" %ith_gen)), alpha=0.2) # second layer
    G_batchnorm2 = tf.layers.batch_normalization(h2, training=False)    
    h3 = tf.nn.leaky_relu(linear(G_batchnorm2, size, ("g_%d_2" %ith_gen)), alpha=0.2) # third layer
    G_batchnorm3 = tf.layers.batch_normalization(h3, training=False)       
    G_prob = tf.nn.softmax(linear(G_batchnorm3, output_size, ("g_%d_3" %ith_gen)))
    return G_prob

############
def alpha_gen(z, z_dim, h_dim, num_gen):
    al_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    al_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    al_W2 = tf.Variable(xavier_init([h_dim, num_gen])) 
    al_b2 = tf.Variable(tf.zeros(shape=[num_gen]))
    theta_G = [al_W1, al_W2, al_b1, al_b2]

    al_h1 = tf.nn.leaky_relu(tf.matmul(z, al_W1) + al_b1, alpha=0.2)        # first layer 
    al_batchnorm = tf.layers.batch_normalization(al_h1, training = False)
    al_log_prob = tf.matmul(al_batchnorm, al_W2) + al_b2                    # second layer
    al_prob = tf.nn.softmax(al_log_prob)
    return al_prob, theta_G
    
############
def discriminator(x, size, h_dim):
    D_W1 = tf.Variable(xavier_init([size, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]

    D_h1 = tf.nn.leaky_relu((tf.matmul(x, D_W1) + D_b1), alpha=0.2) # first layer 
    D_logit = tf.matmul(D_h1, D_W2) + D_b2                       
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit, D_h1, theta_D

############
def gradients_a(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    # Add histograms for variables, gradients and gradient norms in Tensorboard
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + '/gradients', grad_values)
            tf.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )
    return opt.apply_gradients(gradients, global_step=step)

############################################################
class DocModel(object):
    def __init__(self, x, z, params):
        self.x = x
        self.z = z

        ##################### Define Computational Graphs #####################
        ####### Generator init
        self.generators = []
        for i in range(0, params.num_gen):
            with tf.variable_scope('generator_'+str(i), reuse=tf.AUTO_REUSE):
                self.gen_i = generator(z, params.g_i_dim, params.vocab_size, i)                                 # shape=(?, 2000)
                self.generators.append(self.gen_i)
        
        ####### Alpha Generator init
        with tf.variable_scope('alpha_generator', reuse=tf.AUTO_REUSE):
            self.alpha_generator, self.theta_G = alpha_gen(z, params.z_dim, params.d_dim, params.num_gen)       # shape=(?, 5)
        
        ####### Combined Generator init
        with tf.variable_scope('G_sample', reuse=tf.AUTO_REUSE):
            self.G_sample = tf.Variable(tf.zeros(shape=[params.batch_size, params.vocab_size]))                 # shape=(512, 2000)
            for i in range(0, params.num_gen):
                self.G_sample = self.G_sample + self.alpha_generator[:,i][:,np.newaxis] * self.generators[i]
        
        ####### Discriminator init: Real inputs to Discriminator
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            self.D_real, self.D_logit_real, self.rep, self.theta_D = discriminator(x, params.vocab_size, params.d_dim)

        ####### Discriminator init: Fake inputs to Discriminator
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            self.D_fake, self.D_logit_fake, _, _ = discriminator(self.G_sample, params.vocab_size, params.d_dim)


        ##################### Define Losses #####################
        ####### Mean Loss of Discriminator on REAL samples 
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logit_real, \
                                                                                    labels = tf.ones_like(self.D_logit_real)))
        ####### Mean Loss of Discriminator on FAKE samples (G_sample)
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logit_fake, \
                                                                                    labels = tf.zeros_like(self.D_logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        
        ####### Mean Loss of Generator
        self.Gen_loss = []
        for i in range(0, params.num_gen):
            temp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logit_fake, \
                                                                            labels = tf.ones_like(self.D_logit_fake)))
            self.Gen_loss.append(temp)
        
        ####### Mean Loss of Alpha Generator 
        self.Al_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logit_fake, \
                                                                                        labels = tf.ones_like(self.D_logit_fake)))


        ##################### Define Optimizers #####################
        vars = tf.trainable_variables()
        step = tf.Variable(0, trainable=False)
        
        ####### Optimize Discriminator
        self.d_params = [v for v in vars if v.name.startswith('discriminator')]

        self.D_solver = gradients_a(
            opt = tf.train.AdamOptimizer(
                learning_rate = params.learning_rate,
                beta1 = 0.5
            ),
            loss = self.D_loss,
            vars = self.theta_D, 
            step = step
        )

        ####### Optimize Generators
        g_params=[]
        self.G_solver=[]
        
        for i in range(0, params.num_gen):
            g_var = [v for v in vars if v.name.startswith('generator_'+str(i))]
            g_params.append(g_var)
            solver = gradients_a(
                    opt = tf.train.AdamOptimizer(
                        learning_rate = params.learning_rate,
                        beta1 = 0.5
                    ),
                    loss = self.Gen_loss[i],
                    vars = g_var,
                    step = step
                )
            self.G_solver.append(solver)

        ####### Optimize Alpha Generator
        self.al_params = [v for v in vars if v.name.startswith('alpha_generator')]

        self.Al_solver = gradients_a(
            opt = tf.train.AdamOptimizer(
                learning_rate = params.learning_rate,
                beta1 = 0.5
            ),
            loss = self.Al_gen_loss,
            vars = self.theta_G, 
            step = step
        )
