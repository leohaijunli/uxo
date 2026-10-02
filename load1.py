import pandas as pd, numpy as np, glob, os, sys

files = sorted(glob.glob('*-sensys.csv'))
frames=[]
for fn in files:
    d=pd.read_csv(fn)
    t=os.path.basename(fn).replace('-sensys.csv','')
    d['flight']=t
    d['t_sec']=pd.to_datetime(d['Date']+' '+d['Time']).astype('int64')/1e9
    frames.append(d)
df=pd.concat(frames,ignore_index=True)
df=df.dropna(subset=['Latitude','Longitude','TMI']).reset_index(drop=True)
print('files:', files)
print('total rows', len(df), 'flights', df.flight.unique())
print(df[['flight','TMI','Latitude','Longitude']].describe().loc[['min','max','mean','std']])
df.to_pickle(os.path.join(os.environ['TEMP'],'sensys_raw.pkl'))
print('saved')