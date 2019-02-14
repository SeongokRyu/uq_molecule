import tensorflow as tf
from ConcreteDropout import ConcreteDropout
#from tensorflow.keras.layers import Dense, Conv1D

def conv1d_with_concrete_dropout(x, out_dim, wd, dd):
    output = ConcreteDropout(tf.keras.layers.Conv1D(filters=out_dim,
                                                    kernel_size=1,
                                                    use_bias=True,
                                                    activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    bias_initializer=tf.contrib.layers.xavier_initializer()),
                             weight_regularizer=wd,
                             dropout_regularizer=dd,
                             trainable=True )(x, training=True)
    return output

def dense_with_concrete_dropout(x, out_dim, wd, dd):   
    output = ConcreteDropout(tf.keras.layers.Dense(units=out_dim,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                                   bias_initializer=tf.contrib.layers.xavier_initializer()),
                             weight_regularizer=wd,
                             dropout_regularizer=dd,
                             trainable=True)(x, training=True)
    return output

def attn_matrix(A, X, attn_weight):
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F' 
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])
    _X1 = tf.einsum('ij,ajk->aik', attn_weight, tf.transpose(X, [0,2,1]))
    _X2 = tf.matmul(X, _X1)
    _A = tf.multiply(A, _X2)
    _A = tf.nn.tanh(_A)
    return _A

def get_gate_coeff(X1, X2, dim, label):
    num_atoms = int(X1.get_shape()[1])
    _b = tf.get_variable('mem_coef-'+str(label), initializer=tf.contrib.layers.xavier_initializer(), shape=[dim], dtype=tf.float32)
    _b = tf.reshape(tf.tile(_b, [num_atoms]), [num_atoms, dim])

    X1 = tf.layers.dense(X1, units=dim, use_bias=False)
    X2 = tf.layers.dense(X2, units=dim, use_bias=False)
    
    output = tf.nn.sigmoid(X1+X2+_b)
    return output

def graph_attn_gate(A, X, attn, out_dim, label, length, num_train):
    X_total = []
    A_total = []
    wd = length**2/num_train
    dd = 2./num_train
    for i in range( len(attn) ):
        _h = conv1d_with_concrete_dropout(X, out_dim, wd, dd)
        _A = attn_matrix(A, _h, attn[i])
        _h = tf.nn.relu(tf.matmul(_A, _h))
        X_total.append(_h)

    _X = tf.nn.relu(tf.concat(X_total, 2))
    _X = tf.layers.conv1d(_X, 
                          filters=out_dim, 
                          kernel_size=1, 
                          use_bias=False, 
                          activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          bias_initializer=tf.contrib.layers.xavier_initializer())

    dim = int(_X.get_shape()[2])
    if( int(X.get_shape()[2]) != dim ):
        X = tf.layers.dense(X, dim, use_bias=False)
    coeff = get_gate_coeff(_X, X, dim, label)
    output = tf.multiply(_X, coeff) + tf.multiply(X,1.0-coeff)
    return output

def encoder_gat_gate(X, A, num_layers, out_dim, num_attn, length, num_train):
    # X : Atomic Feature, A : Adjacency Matrix
    _X = X
    for i in range(num_layers):
        attn_weight = []
        for j in range( num_attn ):
            attn_weight.append( tf.get_variable('eaw'+str(i)+'_'+str(j), 
                                                initializer=tf.contrib.layers.xavier_initializer(), 
                                                shape=[out_dim, out_dim], 
                                                dtype=tf.float32) 
                              )    

        _X = graph_attn_gate(A, _X, attn_weight, out_dim, i, length, num_train)
    return _X

def readout_and_mlp(X, latent_dim, length, num_train):
    # X : [#Batch, #Atom, #Feature] --> Z : [#Batch, #Atom, #Latent] -- reduce_sum --> [#Batch, #Latent]
    # Graph Embedding in order to satisfy invariance under permutation
    wd = length**2/num_train
    dd = 2./num_train
    Z = tf.nn.relu(conv1d_with_concrete_dropout(X, latent_dim, wd, dd))
    Z = tf.nn.sigmoid(tf.reduce_sum(Z, 1))

    # Predict the molecular property
    _Y = tf.nn.relu(dense_with_concrete_dropout(Z, latent_dim, wd, dd))
    Y_mean = tf.keras.layers.Dense(units=1,
                                   use_bias=True,
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                   bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    Y_logvar = tf.keras.layers.Dense(units=1,
                                     use_bias=True,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                     bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    return Z, Y_mean, Y_logvar
