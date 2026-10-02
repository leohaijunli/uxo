import pandas as pd, numpy as np, os, sys

df = pd.read_pickle(os.path.join(os.environ['TEMP'],'sensys_raw.pkl'))
# compute magnetic anomaly per flight: TMI - regional baseline via large median filter
for fl, grp in df.groupby('flight'):
    idx = grp.index
    # sort by time for continuous line processing
    t = grp['t_sec'].values
    T = grp['TMI'].values
    win = 1001
    base = pd.Series(T).rolling(win, center=True, min_periods=win//2).median().values
    nan = ~np.isfinite(base)
    # fill nan baseline with global flight median
    base[nan] = np.nanmedian(T)
    anom = T - base
    df.loc[idx,'anomaly'] = anom
    df.loc[idx,'baseline'] = base
df.to_pickle(os.path.join(os.environ['TEMP'],'sensys_anom.pkl'))
print('anomaly stats:')
print(df['anomaly'].describe())
print('n with |anom|>50 nT:', (df['anomaly'].abs()>50).sum())