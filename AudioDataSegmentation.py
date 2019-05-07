import numpy as np 
import pandas as pd
from scipy.io import wavfile
from pydub import AudioSegment 
import os
import sys 
from tqdm import tqdm

## put at the same directory at training or testing folder
## type "python AudioDataSegmentation.py {} " to subprocess, replace {} with training or testing

def main():
    def splitWAV(start,end,oldAudio,newAudio,typ):
        start = start*1000
        end = end*1000
        snippet = AudioSegment.from_wav(os.path.join(directory,oldAudio))
        snippet = snippet[start:end]
        if not(os.path.isdir("{}_segment".format(folder))):
            os.mkdir("{}_segment".format(folder))
        if not(os.path.isdir("{}_segment/{}".format(folder,typ))):    
            os.mkdir("{}_segment/{}".format(folder,typ))
        snippet.export(os.path.join("{}_segment/{}".format(folder,typ),newAudio),format="wav")

    folder = sys.argv[1]
    df = pd.read_csv("{}_meta_update.csv".format(folder))
    directory = "{}/sounds".format(folder)
    xmlfiles = os.listdir(folder)
    xmlfiles.remove('sounds')

    report_list = []
    for xmlfile in tqdm(xmlfiles):
        _df = df.loc[(df['xml_file']==xmlfile) & (df['type']=='events')]
        for ii in range(len(_df)):
            newName = _df.iloc[ii]['CLASS_NAME'].split("/")[1]
            typ = _df.iloc[ii]['CLASS_NAME'].split("/")[0]
            st_time = _df.iloc[ii]['STARTSECOND']
            ed_time = _df.iloc[ii]['ENDSECOND']
            oldAudios = os.listdir('{}/sounds'.format(folder))
            soundKey = xmlfile.split(".")[0]
            oldAudios = [ x for x in oldAudios if soundKey in x]
            for sound in oldAudios:
                pth = sound.split('.')[0].split('_')[1]
                outputName = typ+"_"+soundKey+"_"+pth+"_"+newName
                splitWAV(st_time,ed_time,sound,outputName,typ)
                fs,x = wavfile.read(os.path.join("{}_segment/{}".format(folder,typ),outputName))
                report_list.append((outputName,fs,len(x),pth,soundKey,typ))

        _df_bg = df.loc[(df['xml_file']==xmlfile) & (df['type']=='background')]
        for jj in range(len(_df_bg)):
            bg_newName = str(_df_bg.iloc[jj]['xml_id'])+".wav"
            bg_typ = _df_bg.iloc[jj]['CLASS_NAME']
            bg_st_time = _df_bg.iloc[jj]['STARTSECOND']
            bg_ed_time = _df_bg.iloc[jj]['ENDSECOND']
            bg_oldAudios = os.listdir('{}/sounds'.format(folder))
            bg_soundKey = xmlfile.split(".")[0]
            bg_oldAudios = [ y for y in bg_oldAudios if bg_soundKey in y]
            for bg_sound in bg_oldAudios:
                bg_pth = bg_sound.split('.')[0].split('_')[1]
                bg_outputName = bg_typ+"_"+bg_soundKey+"_"+bg_pth+"_"+bg_newName
                splitWAV(bg_st_time,bg_ed_time,bg_sound,bg_outputName,bg_typ)
                fs,y = wavfile.read(os.path.join("{}_segment/{}".format(folder,bg_typ),bg_outputName))
                report_list.append((bg_outputName,fs,len(y),bg_pth,bg_soundKey,bg_typ))


    report = pd.DataFrame(report_list,columns=['NAME','FRAMES_RATE','NO_FRAMES','PITCH','XMLFILE','LABEL'])
    report.to_csv("{}_segment_key.csv".format(folder),index=False)
# 
if __name__ == "__main__":
    main();