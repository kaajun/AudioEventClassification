import xml.etree.ElementTree as ET
import math
import os
import pandas as pd 
import sys
from tqdm import tqdm


## put at the same directory at training or testing folder
## type "python parsexml.py {} " to subprocess, replace {} with training or testing

def main():
    label = sys.argv[1]
    xmlfiles = os.listdir(label)
    xmlfiles.remove('sounds')
    class_data = {}
    for xmlfile in xmlfiles:
        print(xmlfile)
        tree = ET.parse(label+"/"+xmlfile)
        root = tree.getroot()
        for child in root:
            for item in child:
                header_data = {}
                for header in item:
                    header_data[header.tag]=header.text
                class_data[xmlfile+"_"+child.tag+"_"+item.get('idx')] = header_data
    total_df = []
    for row in class_data.keys():
        df = pd.DataFrame([class_data[row]])
        df['xml_file'] = row.split('_')[0]
        df['type'] = row.split('_')[1]
        df['xml_id'] = row.split('_')[2]
        df['KEY'] = row
        total_df.append(df)
    final = pd.concat(total_df)
    final.to_csv('container.csv',index=False)
    final = pd.read_csv('container.csv',index_col=False)
    
    def extract_background_detail(df):
        xmlfiles = list(df['xml_file'].unique())
        df_list = []
        append_list = []
        for xmlfile in tqdm(xmlfiles):
            _xml_df = df.loc[df['xml_file']==xmlfile]
            _xml_df = _xml_df.sort_values('STARTSECOND')
            prev_end = 0
            _df_bg = _xml_df.loc[_xml_df['type']=='background']
            offset = len(_df_bg)
            _xml_wo_bg = _xml_df.loc[_xml_df['type']=='events']
            for ii in range(len(_xml_wo_bg)):
                start_sec = prev_end
                end_sec = _xml_wo_bg.iloc[ii]['STARTSECOND']
                prev_end = _xml_wo_bg.iloc[ii]['ENDSECOND']
                key = _xml_wo_bg.iloc[ii]['KEY']
                append_list.append((1,'background',end_sec,key[0:10]+"background_"+str(ii+offset+1),start_sec,'background',xmlfile,ii+1+offset))   
            bg_df = pd.DataFrame(append_list,columns=['CLASS_ID','CLASS_NAME','ENDSECOND','KEY','STARTSECOND','type','xml_file','xml_id'])
            df_list.append(pd.concat([_xml_wo_bg,bg_df]))

        total = pd.concat(df_list)
        return total

    final = extract_background_detail(final)

    final.to_csv(label+"_meta_update.csv",index=False)


if __name__ == "__main__":
    main();