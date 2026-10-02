import pandas as pd, numpy as np, os

act = pd.read_csv('EO_properties_and_Locations_field2.csv', encoding='utf-8-sig')
act = act[act.ID.notna()].copy()
act['ID']=act['ID'].astype(str).str.strip()
# parse ferrous
def isf(v):
    v=str(v).strip()
    return v=='1'
act['ferrous']=act['Ferrous (1) or NF (0)'].map(isf)
act['row']=act['ID'].str[0]

lon0, lat0 = -96.8571, 36.3531
R=6371000.0
act['x']=R*np.radians(act.Longitude.astype(float)-lon0)*np.cos(np.radians(lat0))
act['y']=R*np.radians(act.Latitude.astype(float)-lat0)

fer=act[act.ferrous].copy()
print('total items',len(act),'ferrous',len(fer))
print(fer.groupby('row').size())
print(fer[['ID','Item','x','y']].to_string())
fer.to_pickle(os.path.join(os.environ['TEMP'],'actual_ferrous.pkl'))
act.to_pickle(os.path.join(os.environ['TEMP'],'actual_all.pkl'))