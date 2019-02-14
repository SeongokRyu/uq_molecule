import tensorflow as tf
import numpy as np
import blocks

class mc_dropout():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.A = tf.placeholder(dtype=tf.float32, shape = [None, FLAGS.max_atoms, FLAGS.max_atoms])
        self.X = tf.placeholder(dtype=tf.float32, shape = [None, FLAGS.max_atoms, 58])
        self.P = tf.placeholder(dtype=tf.float32, shape = [None])

        self.create_network()

    def create_network(self):
        self.A = tf.cast(self.A, tf.float32)
        self.X = tf.cast(self.X, tf.float32)
        self.P = tf.cast(self.P, tf.float32)
        hidden_dim = self.FLAGS.hidden_dim
        latent_dim = self.FLAGS.latent_dim
        length = self.FLAGS.regularization_scale
        num_train = self.FLAGS.num_train
        num_layers = self.FLAGS.num_layers
        num_attn = self.FLAGS.num_attn

        self._X = blocks.encoder_gat_gate(self.X, self.A, num_layers, hidden_dim, num_attn, length, num_train)
        self.Z, self.P_mean, self.P_logvar = blocks.readout_and_mlp(self._X, latent_dim, length, num_train)

        self.loss = None
        if(self.FLAGS.task_type == 'regression'):
            self.loss = self.loss_regression(self.P, self.P_mean, self.P_logvar)
        elif(self.FLAGS.task_type == 'classification'):
            self.loss = self.loss_classification(self.P, self.P_mean)

        self.lr = tf.Variable(0.0, trainable = False)
        self.opt = self.optimizer( self.lr, self.FLAGS.optimizer )
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        tf.train.start_queue_runners(sess=self.sess)
        print ("Network Ready")

    def loss_regression(self, P_truth, P_mean, P_logvar):
        P_truth = tf.reshape(P_truth, shape=[-1])
        P_mean = tf.reshape(P_mean, shape=[-1])
        P_logvar = tf.reshape(P_logvar, shape=[-1])
        P_truth = tf.cast(P_truth, tf.float32)
        P_mean = tf.cast(P_mean, tf.float32)
        P_logvar = tf.cast(P_logvar, tf.float32)

        pred_loss = tf.reduce_mean(0.5*tf.exp(-P_logvar)*(P_truth-P_mean)**2 + P_logvar*0.5)
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        total_loss = pred_loss + reg_loss
        return total_loss

    def loss_classification(self, P_truth, P_mean):
        P_truth = tf.reshape(P_truth, shape=[-1])
        P_mean = tf.reshape(P_mean, shape=[-1])
        P_truth = tf.cast(P_truth, tf.float32)
        P_mean = tf.cast(P_mean, tf.float32)

        pred_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=P_truth, logits=P_mean))
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        total_loss = pred_loss + reg_loss
        return total_loss

    def optimizer(self, lr, opt_type):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.FLAGS.beta1, beta2=self.FLAGS.beta2, epsilon=1e-09)
        return optimizer.minimize(self.loss)

    def train(self, A, X, P):
        opt, P_mean, P_logvar, loss = self.sess.run([self.opt, self.P_mean, self.P_logvar, self.loss], feed_dict = {self.A : A, self.X : X, self.P : P})
        return P_mean, P_logvar, loss

    def test(self, A, X, P):
        P_mean, P_logvar, loss = self.sess.run([self.P_mean, self.P_logvar, self.loss], feed_dict = {self.A : A, self.X : X, self.P : P})
        return P_mean, P_logvar, loss

    def predict(self, A, X):
        P_mean, P_logvar = self.sess.run([self.P_mean, self.P_logvar], feed_dict = {self.A : A, self.X : X})
        return P_mean, P_logvar

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step=global_step)
        print("model saved to '%s'" % (ckpt_path))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate))
