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

CLASS_NUM = 4

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
    
def train_and_save_model(model='xvector'):
    model = define_xvector()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['acc', km.precision(label=1), km.recall(label=0)])
    model.summary()
    callback_list = [
        ModelCheckpoint('checkpoint-{epoch:02d}.h5', monitor='loss', verbose=1, save_best_only=True, period=2), # do the check point each epoch, and save the best model
        ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, min_lr=1e-6), # reducing the learning rate if the val_loss is not improving
        CSVLogger(filename='training_log.csv'), # logger to csv
        EarlyStopping(monitor='loss', patience=5) # early stop if there's no improvment of the loss
    ]
    
    train_data, train_label = get_data(folder='training')
    train_data, train_label = reduce_bg_data(train_data,train_label)
    encoder = LabelBinarizer()
    train_label = encoder.fit_transform(train_label)
    print("Start Training process \n Training data shape {} \n Training label shape {}".format(train_data.shape,train_label.shape))
    model.fit(train_data, train_label, batch_size=32, epochs=25, verbose=1, validation_split=0.2)
    model.save('re_xvector_model.h5')

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


def load_and_predict_model(model='xvector'):
    model = load_model('{}_model.h5'.format(model), custom_objects={'binary_precision': km.precision(label=1), 'binary_recall':km.recall(label=0)})
    model.summary()
    test_data, test_label = get_data(folder='testing')
    encoder = LabelBinarizer()
    test_label = encoder.fit_transform(test_label)
    res = model.evaluate(test_data, test_label )
    print(res)

def load_and_predict_google_audio(model='xvector'):
    model = load_model('{}_model.h5'.format(model), custom_objects={'binary_precision': km.precision(label=1), 'binary_recall':km.recall(label=0)})
    labels = [ x[:-4] for x in os.listdir('google_testing_npy')]
    
    result = []
    for label in labels:
        if label == "background":
            label_class = np.array([1,0,0,0])
        elif label == "glass":
            label_class = np.array([0,1,0,0])
        elif label == "gunshot":
            label_class = np.array([0,0,1,0])
        else:
            label_class = np.array([0,0,0,1])
        print("Start prediction for {}".format(label))
        directory = os.path.join('google_testing_npy',"{}_npy".format(label))
        npys = os.listdir(directory)
        for npy in tqdm(npys):
            data = np.transpose(np.load(os.path.join("google_testing_npy/{}_npy".format(label),npy)))
            data = data[newaxis,:,:]
            pred = model.predict(data)
            pred = np.ravel(pred.astype('int32'))
            result.append((npy,pred,label_class,np.array_equal(pred,label_class)))
    

    report =  pd.DataFrame(result,columns=['NAME','PREDICTION','LABEL','RESULT'])
    report.to_csv('google_test_result.csv',index=False)
    print("The accuracy of the model is {}".format(str(float(report['RESULT'].sum())/len(report))))

def load_and_predict_google_audio2(model='re_xvector'):
    model = load_model('{}_model.h5'.format(model), custom_objects={'binary_precision': km.precision(label=1), 'binary_recall':km.recall(label=0)})
    labels = [ x[:-4] for x in os.listdir('google_testing_npy')]
    encoder = LabelBinarizer()
    _ = encoder.fit_transform(np.array([1,2,3,4]))
    result = []
    y_pred = []
    y_target = []
    for label in labels:
        if label == "background":
            label_class = np.array([1])
        elif label == "glass":
            label_class = np.array([2])
        elif label == "gunshot":
            label_class = np.array([3])
        else:
            label_class = np.array([4])
        print("Start prediction for {}".format(label))
        directory = os.path.join('google_testing_npy',"{}_npy".format(label))
        npys = os.listdir(directory)
        window = 25
        stride = 10
        
        for npy in tqdm(npys):
            _seg_data = []
            _seg_label = []
            npy_array = np.transpose(np.load(os.path.join("google_testing_npy/{}_npy".format(label),npy)))
            #data = data[newaxis,:,:]
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
            _seg_label = encoder.transform(_seg_label)
            pred = model.predict(_seg_data)
            pred = np.mean(pred, axis=0)
            _seg_label = np.mean(_seg_label, axis=0)
            y_pred.append(pred)
            y_target.append(_seg_label)
            result.append((npy,pred,np.around(pred),_seg_label,np.array_equal(np.around(pred),_seg_label)))

    
    report =  pd.DataFrame(result,columns=['NAME','PREDICTION1','PREDICTION','LABEL','RESULT'])
    report.to_csv('google_test_result2.csv',index=False)
    y_target = encoder.inverse_transform(np.array(y_target))
    y_predicted = encoder.inverse_transform(np.array(y_pred))
    cm = confusion_matrix(y_target=y_target,y_predicted=y_predicted,binary=False)
    fig,ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()
    plt.savefig('confusion_matrix_google_re_xvector.png')
    print("The accuracy of the model is {}".format(str(float(report['RESULT'].sum())/len(report))))

def load_and_test_wo_bg(model='xvector'):
    #model = load_model('{}_model.h5'.format(model), custom_objects={'binary_precision': km.precision(label=1), 'binary_recall':km.recall(label=0)})
    npys = os.listdir('testing_npy')
    labels = [ x.split("_")[0] for x in npys ]
    labels = unique(labels).tolist()
    labels.remove('background')
    npy_list = []
    for npy in npys:
        if (labels[0] in npy) or (labels[1] in npy) or (labels[2] in npy):
            npy_list.append(npy)
    



if __name__ == "__main__":
    load_and_test_wo_bg()
    
