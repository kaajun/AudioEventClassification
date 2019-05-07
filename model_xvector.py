from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

import os
import keras.backend as K
import keras_metrics as km
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *

from math import floor
from tqdm import tqdm
from speechpy import speechpy
from sklearn.preprocessing import LabelBinarizer
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

CLASS_NUM = 1

def apply_cmvn(npy):
    return speechpy.processing.cmvnw(npy)


def mean_std_pool1d(x):
    mean = K.mean(x, axis=1)
    std = K.std(x, axis=1)
    return K.concatenate([mean, std], axis=1) 
def mean_std_pool1d_output_shape(input_shape):
    shape = list(input_shape)
    return tuple([shape[0], shape[2]*2])

def get_data(folder='training',window=25,stride=10):
    directory = '{}_npy'.format(folder)
    npys = os.listdir(directory)
    if not(os.path.isdir('{}_fin_data'.format(folder))):
        os.mkdir('{}_fin_data'.format(folder))
    if not(os.path.isdir('{}_fin_label'.format(folder))):
        os.mkdir('{}_fin_label'.format(folder))
    _data = []
    _label = []
    if not (os.path.isfile("{}_data_npy.npy".format(folder)) and os.path.isfile("{}_label_npy.npy".format(folder))):
        for npy in tqdm(npys):
            _seg_data = []
            _seg_label = []
            if not(os.path.isfile("{0}_fin/{1}_data.npy".format(folder,npy)) and os.path.isfile("{0}_fin/{1}_label.npy".format(folder,npy))):
                lb = npy.split("_")[0]
                if lb == 'background':
                    label_class = np.array([1])
                elif lb == 'glass':
                    label_class = np.array([2])
                elif lb == 'gunshots':
                    label_class = np.array([3])
                else:
                    label_class = np.array([4])
                npy_array = np.load(os.path.join(directory,npy)).transpose()
                npy_array = apply_cmvn(npy_array)
                dt_points = floor((npy_array.shape[0]-window)/stride) + 1
                for ii in range(dt_points):
                    _seg_array = npy_array[ii*stride:ii*stride+window,:]
                    _seg_array = _seg_array[newaxis,:,:]
                    if len(_seg_data)==0 :
                        _seg_data = _seg_array
                        _seg_label = label_class
                    else:
                        _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
                        _seg_label = np.concatenate([_seg_label,label_class],axis=0)
                # last piece of array
                _seg_array = npy_array[npy_array.shape[0]-window:npy_array.shape[0],:]
                _seg_array = _seg_array[newaxis,:,:]
                _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
                _seg_label = np.concatenate([_seg_label,label_class],axis=0)
                np.save("{0}_fin_data/{1}_data.npy".format(folder,npy),_seg_data)
                np.save("{0}_fin_label/{1}_label.npy".format(folder,npy),_seg_label)
            else:
                _seg_data = np.load("{0}_fin_data/{1}_data.npy".format(folder,npy))
                _seg_label = np.load("{0}_fin_label/{1}_label.npy".format(folder,npy))
            if len(_data) == 0:
                _data = _seg_data
                _label = _seg_label
            else:
                _data = np.concatenate([_data,_seg_data],axis=0)
                _label = np.concatenate([_label,_seg_label],axis=0)
        np.save("{}_data_npy.npy".format(folder),_data)
        np.save("{}_label_npy.npy".format(folder),_label)
    else:
        _data = np.load("{}_data_npy.npy".format(folder))
        print("{} data loaded!".format(folder))        
        _label = np.load("{}_label_npy.npy".format(folder))
        print("{} label loaded!".format(folder))
    return _data,_label




def define_xvector():
    input_layer = Input(shape=(None, 140))
    # xvector model
    x = Conv1D(512, 3, padding='causal', name='tdnn_1')(input_layer)
    x = Activation('relu', name='tdnn_act_1')(x)
    x = Conv1D(512, 3, padding='causal', name='tdnn_2')(x)
    x = Activation('relu', name='tdnn_act_2')(x)
    x = Conv1D(512, 3, padding='causal', name='tdnn_3')(x)
    x = Activation('relu', name='tdnn_act_3')(x)
    x = Conv1D(512, 1, padding='causal', name='tdnn_4')(x)
    x = Activation('relu', name='tdnn_act_4')(x)
    x = Conv1D(1500, 1, padding='causal', name='tdnn_5')(x)
    x = Activation('relu', name='tdnn_act_5')(x)
    x = Lambda(mean_std_pool1d, output_shape=mean_std_pool1d_output_shape, name='stats_pool')(x)
    # fully-connected network
    x = Dense(512, activation='relu', name='feature_layer')(x)
    x = Dense(512, activation='relu', name='fc_2')(x)
    x = Dense(CLASS_NUM, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


def define_model_etdnn():
    input_layer = Input(shape=(None, 140))

    # extended TDNN model with Skip Connection
    x1 = Conv1D(512, 3, padding='causal', name='tdnn_1')(input_layer)
    x1 = Activation('relu', name='tdnn_act_1')(x1)
    x2 = Dense(512, name='tdnn_2')(x1)
    x2 = Activation('relu', name='tdnn_act_2')(x2)
    x3 = Conv1D(512, 3, padding='causal', name='tdnn_3')(x2)
    x3 = Activation('relu', name='tdnn_act_3')(x3)

    # skip 1 to 3
    x = Concatenate(name='skip_conn_1')([x3,x1])
    x4 = Dense(512, name='tdnn_4')(x)
    x4 = Activation('relu', name='tdnn_act_4')(x4)
    x5 = Conv1D(512, 4, padding='causal', name='tdnn_5')(x4)
    x5 = Activation('relu', name='tdnn_act_5')(x5)

    # skip 1 to 5 and 3 to 5
    x = Concatenate(name='skip_conn_2')([x5, x3, x1])
    x6 = Dense(512, name='tdnn_6')(x)
    x6 = Activation('relu', name='tdnn_act_6')(x6)
    x7 = Dense(512, name='tdnn_7')(x6)
    x7 = Activation('relu', name='tdnn_act_7')(x7)

    # skip 3 to 7 and 5 to 7
    x = Concatenate(name='skip_conn_3')([x7, x5, x3])

    # statistical pooling
    x = Lambda(mean_std_pool1d, output_shape=mean_std_pool1d_output_shape, name='stats_pool')(x)

    # fully-connected network
    x = Dense(512, activation='relu', name='feature_layer')(x)
    x = Dense(512, activation='relu', name='fc_2')(x)
    x = Dense(CLASS_NUM, activation='softmax', name='output_layer')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model
    
def train_and_save_model(model='xvector',binary_class=False,single_class='glass'):
    model = define_xvector()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['acc', km.precision(label=1), km.recall(label=0)])
    #model.summary()
    callback_list = [
        ModelCheckpoint('checkpoint-{epoch:02d}.h5', monitor='loss', verbose=1, save_best_only=True, period=2), # do the check point each epoch, and save the best model
        ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, min_lr=1e-6), # reducing the learning rate if the val_loss is not improving
        CSVLogger(filename='training_log.csv'), # logger to csv
        EarlyStopping(monitor='loss', patience=5) # early stop if there's no improvment of the loss
    ]
    
    train_data, train_label = get_data(folder='training')
    if binary_class == False:
        train_data, train_label = reduce_bg_data(train_data,train_label)
    else:
        train_data,train_label = filter_reduce_bg(train_data,train_label,single_class)
    encoder = LabelBinarizer()
    train_label = encoder.fit_transform(train_label)
    print("Start Training process \nTraining data shape {} \nTraining label shape {}".format(train_data.shape,train_label.shape))
    model.fit(train_data, train_label, batch_size=32, epochs=25, verbose=1, validation_split=0.2)
    if binary_class == False:
        model.save('re_{}_model.h5'.format(model))
    else:
        model.save('re_{}_{}.h5'.format(model,single_class))

def reduce_bg_data(data,label,percent=0.3,folder='training'):
    if not (os.path.isfile("re_{}_data_npy.npy".format(folder)) and os.path.isfile("re_{}_label_npy.npy".format(folder))):
        df = pd.DataFrame(label,columns=['label'])
        df['data'] = [ x for x in data]
        #print(df['label'].value_counts())
        df1 = df.loc[df['label']==1]
        df1 = df1.sample(frac=percent)
        df_total = pd.concat([df1,df.loc[df['label']!=1]])
        data = np.array(df_total['data'].tolist())
        label = df_total['label'].values
        np.save("re_{}_data_npy.npy".format(folder),data)
        np.save("re_{}_label_npy.npy".format(folder),label)
    else:
        data = np.load("re_{}_data_npy.npy".format(folder))     
        label = np.load("re_{}_label_npy.npy".format(folder))
    return data, label

def filter_reduce_bg(data,label,classFilter,percent=0.1,folder='training'):
    if classFilter == 'glass':
        label_filter = 2
    elif classFilter == 'gunshots':
        label_filter = 3
    else:
        label_filter = 4
    if not (os.path.isfile("re_{}_data_npy.npy".format(classFilter)) and os.path.isfile("re_{}_label_npy.npy".format(classFilter))):
        df = pd.DataFrame(label,columns=['label'])
        df['data'] = [ x for x in data]
        #print(df['label'].value_counts())
        df1 = df.loc[df['label']==1]
        df1 = df1.sample(frac=percent)
        df_total = pd.concat([df1,df.loc[df['label']==label_filter]])
        data = np.array(df_total['data'].tolist())
        label = df_total['label'].values
        np.save("re_{}_data_npy.npy".format(classFilter),data)
        np.save("re_{}_label_npy.npy".format(classFilter),label)
    else:
        data = np.load("re_{}_data_npy.npy".format(classFilter))     
        label = np.load("re_{}_label_npy.npy".format(classFilter))
    un, ct = np.unique(label, return_counts=True)
    label_count = dict(zip(un,ct))
    print("Bi-class training initiated! \nBackground training data = {}\n{} training data = {}".format(label_count[1],classFilter,label_count[label_filter]))
    return data, label




def custom_test(model='re_xvector'):
    window = 25
    stride =10
    npy = np.load("testing_npy/glass_00003_1_0035.npy")
    model = load_model('{}_model.h5'.format(model), custom_objects={'binary_precision': km.precision(label=1), 'binary_recall':km.recall(label=0)})
    encoder = LabelBinarizer()
    _ = encoder.fit_transform(np.array([1,2,3,4]))
    _seg_data = []
    _seg_label = []
    npy_array = np.transpose(npy)
    npy_array = apply_cmvn(npy_array)
    dt_points = floor((npy_array.shape[0]-window)/stride) + 1
    for ii in range(dt_points):
        _seg_array = npy_array[ii*stride:ii*stride+window,:]
        _seg_array = _seg_array[newaxis,:,:]
        if len(_seg_data)==0 :
            _seg_data = _seg_array
            #_seg_label = label
        else:
            _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
            #_seg_label = np.concatenate([_seg_label,label],axis=0)
        # last piece of array
    _seg_array = npy_array[npy_array.shape[0]-window:npy_array.shape[0],:]
    _seg_array = _seg_array[newaxis,:,:]
    _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
    #_seg_label = np.concatenate([_seg_label,label],axis=0)

    #_seg_label = encoder.transform(_seg_label)
    pred = model.predict(_seg_data)
    print(pred)
    np.save("pred_glass_00003_1_0035.npy",pred)

def load_and_test(model='re_xvector',bg_filter=True,segmentize=True,testset='mivia'):
    mstr = model
    model = load_model('{}_model.h5'.format(model), custom_objects={'binary_precision': km.precision(label=1), 'binary_recall':km.recall(label=0)})
    encoder = LabelBinarizer()
    _ = encoder.fit_transform(np.array([1,2,3,4]))
    if testset == 'mivia':
        npys = os.listdir('testing_npy')
        direc = 'testing_npy'
    else:
        npys = os.listdir('google_testing_npy')
        direc = 'google_testing_npy'
    labels = [ x.split("_")[0] for x in npys ]
    labels = unique(labels).tolist()
    npy_list = []
    result = []
    y_target = []
    y_predicted = []
    window = 25
    stride = 10
    if bg_filter == True:
        bgstr = 'wobg'
        try:
            labels.remove('background')
        except:
            print("doens't have background data")
        for npy in npys:
            if (labels[0] in npy) or (labels[1] in npy) or (labels[2] in npy):
                npy_list.append(npy)
    else:
        bgstr = 'withbg'
        npy_list = npys
         
    for npy in tqdm(npy_list):
        directory = os.path.join(direc,npy)
        npy_array = np.transpose(np.load(directory))
        npy_array = apply_cmvn(npy_array)
        class_label = npy.split("_")[0]
        if class_label == 'background':
            label = np.array([1])
        elif class_label == 'glass':
            label = np.array([2])
        elif class_label == 'gunshots':
            label = np.array([3])
        else:
            label = np.array([4])
        if segmentize == True:
            sgstr = 'seg'
            _seg_data = []
            _seg_label = []
            dt_points = floor((npy_array.shape[0]-window)/stride) + 1
            for ii in range(dt_points):
                _seg_array = npy_array[ii*stride:ii*stride+window,:]
                _seg_array = _seg_array[newaxis,:,:]
                if len(_seg_data)==0 :
                    _seg_data = _seg_array
                    _seg_label = label
                else:
                    _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
                    _seg_label = np.concatenate([_seg_label,label],axis=0)
            # last piece of array
            _seg_array = npy_array[npy_array.shape[0]-window:npy_array.shape[0],:]
            _seg_array = _seg_array[newaxis,:,:]
            # join all together
            _seg_data = np.concatenate([_seg_data,_seg_array],axis=0)
            _seg_label = np.concatenate([_seg_label,label],axis=0)
            _seg_label = encoder.transform(_seg_label)
            pred = model.predict(_seg_data)
            pred = np.mean(pred, axis=0)
            pred = np.ravel(pred)
            _seg_label = np.mean(_seg_label, axis=0)
        else:
            sgstr = 'noseg'
            npy_array = npy_array[newaxis,:,:]
            pred = model.predict(npy_array)
            pred = np.ravel(pred)
            _seg_label = encoder.transform(label)
            _seg_label = np.ravel(_seg_label)
        a = [0,0,0,0]
        idx = np.argmax(pred)
        a[idx] = 1
        pred_r = np.array(a)
        y_predicted.append(pred_r)
        y_target.append(_seg_label)
        result.append((npy,pred,pred_r,_seg_label,np.array_equal(np.argmax(pred),np.argmax(_seg_label))))
    
    report =  pd.DataFrame(result,columns=['NAME','PREDICTION1','PREDICTION','LABEL','RESULT'])
    
    report.to_csv('{}_{}_test_{}_{}.csv'.format(bgstr,testset,mstr,sgstr),index=False)
    y_target = encoder.inverse_transform(np.array(y_target))
    y_predicted = encoder.inverse_transform(np.array(y_predicted))
    cm = confusion_matrix(y_target=y_target,y_predicted=y_predicted,binary=False)
    fig,ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()
    plt.savefig('{}_confusion_matrix_{}_{}_{}.png'.format(bgstr,testset,mstr,sgstr))
    print("The accuracy of the model is {}".format(str(float(report['RESULT'].sum())/len(report))))
    



if __name__ == "__main__":
    load_and_test(model='re_xvector',bg_filter=False,segmentize=False,testset='google')
    
