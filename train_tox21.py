import numpy as np
import os
import time
import sys
import random
from utils import shuffle_two_list, load_input_tox21, convert_to_graph, split_train_eval_test
from rdkit import Chem
from mc_dropout import mc_dropout
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
np.set_printoptions(precision=3)

def np_sigmoid(x):
    return 1./(1.+np.exp(-x))

def training(model, FLAGS, model_name, smi_total, prop_total):
    print ("Start Training XD")
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    init_lr = FLAGS.init_lr
    total_st = time.time()
    smi_train, smi_eval, smi_test = split_train_eval_test(smi_total, 0.8, 0.2, 0.1)
    prop_train, prop_eval, prop_test = split_train_eval_test(prop_total, 0.8, 0.2, 0.1)
    prop_eval = np.asarray(prop_eval)
    prop_test = np.asarray(prop_test)
    num_train = len(smi_train)
    num_eval = len(smi_eval)
    num_test = len(smi_test)
    smi_train = smi_train[:num_train]
    prop_train = prop_train[:num_train]
    num_batches_train = (num_train//batch_size) + 1
    num_batches_eval = (num_eval//batch_size) + 1
    num_batches_test = (num_test//batch_size) + 1
    num_sampling = 20
    total_iter = 0
    print("Number of-  training data:", num_train, "\t evaluation data:", num_eval, "\t test data:", num_test)
    for epoch in range(num_epochs):
        st = time.time()
        lr = init_lr * 0.5**(epoch//10)
        model.assign_lr(lr)
        smi_train, prop_train = shuffle_two_list(smi_train, prop_train)
        prop_train = np.asarray(prop_train)

        # TRAIN
        num = 0
        train_loss = 0.0
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        for i in range(num_batches_train):
            num += 1
            st_i = time.time()
            total_iter += 1
            A_batch, X_batch = convert_to_graph(smi_train[i*batch_size:(i+1)*batch_size], FLAGS.max_atoms) 
            Y_batch = prop_train[i*batch_size:(i+1)*batch_size]

            Y_mean, _, loss = model.train(A_batch, X_batch, Y_batch)
            train_loss += loss
            Y_pred = np_sigmoid(Y_mean.flatten())
            Y_pred_total = np.concatenate((Y_pred_total, Y_pred), axis=0)
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)

            et_i = time.time()

        train_loss /= num
        train_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
        train_auroc = 0.0
        try:
            train_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
        except:
            train_auroc = 0.0    

        #Eval
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        num = 0
        eval_loss = 0.0
        for i in range(num_batches_eval):
            A_batch, X_batch = convert_to_graph(smi_eval[i*batch_size:(i+1)*batch_size], FLAGS.max_atoms) 
            Y_batch = prop_eval[i*batch_size:(i+1)*batch_size]
        
            # MC-sampling
            P_mean = []
            for n in range(3):
                num += 1
                Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
                eval_loss += loss
                P_mean.append(Y_mean.flatten())

            P_mean = np_sigmoid(np.asarray(P_mean))
            mean = np.mean(P_mean, axis=0)
    
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
            Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)

        eval_loss /= num
        eval_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
        eval_auroc = 0.0
        try:
            eval_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
        except:
            eval_auroc = 0.0    

        # Save network! 
        ckpt_path = 'save/'+model_name+'.ckpt'
        model.save(ckpt_path, epoch)
        et = time.time()
        # Print Results
        print ("Time for", epoch, "-th epoch: ", et-st)
        print ("Loss        Train:", round(train_loss,3), "\t Evaluation:", round(eval_loss,3))
        print ("Accuracy    Train:", round(train_accuracy,3), "\t Evaluation:", round(eval_accuracy,3))
        print ("AUROC       Train:", round(train_auroc,3), "\t Evaluation:", round(eval_auroc,3))
    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))

    #Test
    test_st = time.time()
    Y_pred_total = np.array([])
    Y_batch_total = np.array([])
    ale_unc_total = np.array([])
    epi_unc_total = np.array([])
    tot_unc_total = np.array([])
    num = 0
    test_loss = 0.0
    for i in range(num_batches_test):
        num += 1
        A_batch, X_batch = convert_to_graph(smi_test[i*batch_size:(i+1)*batch_size], FLAGS.max_atoms) 
        Y_batch = prop_test[i*batch_size:(i+1)*batch_size]
        
        # MC-sampling
        P_mean = []
        for n in range(num_sampling):
            Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
            P_mean.append(Y_mean.flatten())

        P_mean = np_sigmoid(np.asarray(P_mean))

        mean = np.mean(P_mean, axis=0)
        ale_unc = np.mean(P_mean*(1.0-P_mean), axis=0)
        epi_unc = np.mean(P_mean**2, axis=0) - np.mean(P_mean, axis=0)**2
        tot_unc = ale_unc + epi_unc
    
        Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
        Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)
        ale_unc_total = np.concatenate((ale_unc_total, ale_unc), axis=0)
        epi_unc_total = np.concatenate((epi_unc_total, epi_unc), axis=0)
        tot_unc_total = np.concatenate((tot_unc_total, tot_unc), axis=0)

    np.save('./statistics/'+model_name+'_mc_truth.npy', Y_batch_total)
    np.save('./statistics/'+model_name+'_mc_pred.npy', Y_pred_total)
    np.save('./statistics/'+model_name+'_mc_epi_unc.npy', epi_unc_total)
    np.save('./statistics/'+model_name+'_mc_ale_unc.npy', ale_unc_total)
    np.save('./statistics/'+model_name+'_mc_tot_unc.npy', tot_unc_total)
    test_et = time.time()
    print ("Finish Testing, Total time for test:", (test_et-test_st))
    return

dim1 = 32
dim2 = 256
max_atoms = 75
num_layer = 4
batch_size = 256
epoch_size = 100
learning_rate = 0.001
regularization_scale = 1e-4
beta1 = 0.9
beta2 = 0.98
prop = sys.argv[1]

smi_total, prop_total = load_input_tox21(prop, max_atoms)
num_total = len(smi_total)
num_test = int(num_total*0.2)
num_train = num_total-num_test
num_eval = int(num_train*0.1)
num_train -= num_eval


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set FLAGS for environment setting
flags = tf.app.flags
FLAGS = flags.FLAGS
# Hyperparameters for a transfer-trained model
flags.DEFINE_string('task_type', 'classification', '')
flags.DEFINE_integer('hidden_dim', dim1, '')
flags.DEFINE_integer('latent_dim', dim2, '')
flags.DEFINE_integer('max_atoms', max_atoms, '')
flags.DEFINE_integer('num_layers', num_layer, '# of hidden layers')
flags.DEFINE_integer('num_attn', 4, '# of heads for multi-head attention')
flags.DEFINE_integer('batch_size', batch_size, 'Batch size')
flags.DEFINE_integer('epoch_size', epoch_size, 'Epoch size')
flags.DEFINE_integer('num_train', num_train, 'Number of training data')
flags.DEFINE_float('regularization_scale', regularization_scale, '')
flags.DEFINE_float('beta1', beta1, '')
flags.DEFINE_float('beta2', beta2, '')
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp') 
flags.DEFINE_float('init_lr', learning_rate, 'Batch size')

model_name = 'MC-Dropout_tox21_'+prop
print("Do Single-Task Learning")
print("Hidden dimension of graph convolution layers:", dim1)
print("Hidden dimension of readout & MLP layers:", dim2)
print("Maximum number of allowed atoms:", max_atoms)
print("Batch sise:", batch_size, "Epoch size:", epoch_size)
print("Initial learning rate:", learning_rate, "\t Beta1:", beta1, "\t Beta2:", beta2, "for the Adam optimizer used in this training")

model = mc_dropout(FLAGS)
training(model, FLAGS, model_name, smi_total, prop_total)
