import os
import sys
import librosa
import librosa.display
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy.io import wavfile 
from speechpy import speechpy

def main():
    #folder = sys.argv[1]
    folder = 'testing'
    def apply_cmvn(npy):
        return speechpy.processing.cmvnw(npy)

    def extract_audio_training(wav,directory,SAMPLE_RATE=16000,FRAME_LENGTH=400,STRIDE=160,n_mels=60,n_mfcc=20,width=9,n_chroma=13,n_bands=6,tograph=False):
        _clip, _sr = librosa.load(os.path.join(directory,wav),sr=SAMPLE_RATE)
        _stft = librosa.stft(_clip,n_fft=FRAME_LENGTH,hop_length=STRIDE)
        _mels = librosa.feature.melspectrogram(_clip, n_fft=FRAME_LENGTH, hop_length=STRIDE, n_mels=n_mels, sr=SAMPLE_RATE)
        _mfcc = librosa.feature.mfcc(y=_clip,sr=SAMPLE_RATE,n_mfcc=n_mfcc,n_fft=FRAME_LENGTH,hop_length=STRIDE)
        _mfcc_d = librosa.feature.delta(_mfcc,width=width,order=1)
        _mfcc_dd = librosa.feature.delta(_mfcc,width=width,order=2)
        _chroma = librosa.feature.chroma_stft(y=_clip,sr=SAMPLE_RATE,n_fft=FRAME_LENGTH,hop_length=STRIDE,n_chroma=n_chroma)
        _spec_cons = librosa.feature.spectral_contrast(y=_clip,sr=SAMPLE_RATE,n_fft=FRAME_LENGTH,hop_length=STRIDE, n_bands=n_bands)
        _res = np.concatenate((_mels,_mfcc,_mfcc_d,_mfcc_dd,_chroma,_spec_cons),axis=0)
        _time_step = _res.shape[1]
        _res = apply_cmvn(_res)
        if (tograph):
            fig, ax = plt.subplots(1,7, figsize=(30,6))
            librosa.display.specshow(_stft, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE, ax=ax[0])
            ax[0].set_title('Short-Time Fourier Transform \n n_fft={}, hop_length={}, \n time_steps={}, fft_bins={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_stft.shape[1],_stft.shape[0],_stft.shape))
            librosa.display.specshow(_mels, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE,ax=ax[1])
            ax[1].set_title('Mels Spectogram \n n_fft={}, hop_length={}, \n time_steps={}, n_mels={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_mels.shape[1],_mels.shape[0],_mels.shape))
            librosa.display.specshow(_mfcc, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE,ax=ax[2])
            ax[2].set_title('MFCC \n n_fft={}, hop_length={}, \n time_steps={}, n_mfcc={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_mfcc.shape[1],_mfcc.shape[0],_mfcc.shape))
            librosa.display.specshow(_mfcc_d, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE,ax=ax[3])
            ax[3].set_title('MFCC_D \n n_fft={}, hop_length={}, \n time_steps={}, n_mfcc={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_mfcc_d.shape[1],_mfcc_d.shape[0],_mfcc_d.shape))
            librosa.display.specshow(_mfcc_dd, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE,ax=ax[4])
            ax[4].set_title('MFCC_DD \n n_fft={}, hop_length={}, \n time_steps={}, n_mfcc={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_mfcc_dd.shape[1],_mfcc_dd.shape[0],_mfcc_dd.shape))
            librosa.display.specshow(_chroma, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE,ax=ax[5])
            ax[5].set_title('Chroma_STFT \n n_fft={}, hop_length={}, \n time_steps={}, n_chroma={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_chroma.shape[1],_chroma.shape[0],_chroma.shape))
            librosa.display.specshow(_spec_cons, x_axis='time',y_axis='linear',sr=SAMPLE_RATE,hop_length=STRIDE,ax=ax[6])
            ax[6].set_title('Spectral Constrast \n n_fft={}, hop_length={}, \n time_steps={}, n_fband={} \n (2D resulting shape: {})'.format(FRAME_LENGTH,STRIDE,_spec_cons.shape[1],_spec_cons.shape[0],_spec_cons.shape))
            #plt.subplots_adjust(wspace=0.45)
            fig.tight_layout()
            plt.savefig("STFT vs Mels vs MFCC and Ds vs Chroma vs SC.png")
            plt.show()
        return _res,_time_step


    if not(os.path.isdir('google_{}_npy'.format(folder))):
        os.mkdir('google_{}_npy'.format(folder))

    

    #df = pd.read_csv("{}_segment_key.csv".format(folder))
    #labels = list(df['LABEL'].unique())
    labels = [x[:-4] for x in os.listdir('google_audioset')]
    report_list = []

    
    for label in labels:

        directory = "google_audioset/{}_WAV".format(label)
        wavs = os.listdir(directory)
        if not(os.path.isdir('google_{}_npy/{}_npy'.format(folder,label))):
            os.mkdir('google_{}_npy/{}_npy'.format(folder,label))
        ouputDir = 'google_{}_npy/{}_npy'.format(folder,label)
        print("Start extraction of class : ",label)
        for wav in tqdm(wavs):
            wav_npy, ts =extract_audio_training(wav,directory)
            np.save(os.path.join(ouputDir,label+'_'+wav[0:12]+'.npy'),wav_npy)
            report_list.append((label+'_'+wav[0:12]+'.npy',ts,label))

    report =  pd.DataFrame(report_list,columns=['NAME','TIME_STEP','LABEL'])
    report.to_csv('google_{}_time_step.csv'.format(folder),index=False)


if __name__ == "__main__" :
    main()






