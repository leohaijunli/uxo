import pandas as pd, numpy as np, os
from scipy.ndimage import maximum_filter, label, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.getcwd(),'report_output')
os.makedirs(OUT, exist_ok=True)
TEMP = os.environ['TEMP']

# ---------- load ----------
df = pd.read_pickle(os.path.join(TEMP,'sensys_anom.pkl'))
df = df[(df.Latitude.between(36.35298,36.35320)) & (df.Longitude.between(-96.85730,-96.85685))].copy()
lon0, lat0 = -96.8571, 36.3531
R = 6371000.0
df['x']=R*np.radians(df.Longitude.values-lon0)*np.cos(np.radians(lat0))
df['y']=R*np.radians(df.Latitude.values-lat0)
act = pd.read_pickle(os.path.join(TEMP,'actual_all.pkl'))
act['x']=R*np.radians(act.Longitude.astype(float)-lon0)*np.cos(np.radians(lat0))
act['y']=R*np.radians(act.Latitude.astype(float)-lat0)
targets = act[act.ferrous & act['row'].isin(list('ABCDE'))].copy()
allfer  = act[act.ferrous].copy()

# ---------- grid ----------
cell=0.20
gx = np.arange(np.floor(df.x.min()/cell), np.ceil(df.x.max()/cell)+1)*cell
gy = np.arange(np.floor(df.y.min()/cell), np.ceil(df.y.max()/cell)+1)*cell
Xi = np.clip(np.floor((df.x-df.x.min())/cell).astype(int),0,len(gx)-1)
Yi = np.clip(np.floor((df.y-df.y.min())/cell).astype(int),0,len(gy)-1)
summ=np.zeros((len(gy),len(gx))); cnt=np.zeros((len(gy),len(gx)))
np.add.at(summ,(Yi,Xi),df.anomaly.values); np.add.at(cnt,(Yi,Xi),1)
cnt[cnt==0]=1; anom=summ/cnt
amap=np.abs(anom)

# analytic signal (magnitude of the gradient): one peak per source
sigma_as, msz_as, pct_as, minsep_as = 0.5, 5, 80, 0.9
anom_g = gaussian_filter(anom, sigma=sigma_as)
gy_, gx_ = np.gradient(anom_g, gy, gx, edge_order=1)
asig = np.sqrt(gx_**2 + gy_**2)
thr = np.percentile(asig[asig>0], pct_as)
mx = maximum_filter(asig, size=msz_as)
pk = (asig==mx) & (asig > thr)
lbl, nf = label(pk)
cands=[]
for k in range(1,nf+1):
    ys,xs=np.where(lbl==k); w=asig[ys,xs]
    cands.append(( (gx[xs]*w).sum()/w.sum(), (gy[ys]*w).sum()/w.sum(), w.max() ))
cands=np.array(cands)
keep=[]
for o in np.argsort(-cands[:,2]):
    c=cands[o]
    if len(keep):
        k=np.array(keep)
        if np.min(np.sqrt((k[:,0]-c[0])**2+(k[:,1]-c[1])**2))<minsep_as: continue
    keep.append(c)
peaks_xy=np.array(keep)
np.save(os.path.join(TEMP,'peaks_xy.npy'),peaks_xy)
print('analytic-signal detected peaks:', len(peaks_xy))

# ---------- match (one-to-one Hungarian assignment) ----------
from scipy.optimize import linear_sum_assignment
tx = targets.x.values; ty = targets.y.values
px = peaks_xy[:,0];   py = peaks_xy[:,1]
# distance matrix targets x peaks
D = np.sqrt((tx[:,None]-px[None,:])**2 + (ty[:,None]-py[None,:])**2)
rows=[]
used_peak=set()
# one-to-one: assign each target a distinct nearest unused peak (greedy by error), else mark missed
order = np.argsort(D.min(axis=1))          # process targets from best-match first
for ti in order:
    ds = D[ti]
    # candidate peaks not yet used, sorted by distance
    cand = np.argsort(ds)
    j = None
    for jj in cand:
        if jj not in used_peak:
            j = jj; break
    m = ds[j]
    if m < 2.0:
        used_peak.add(j)
        rows.append(dict(ID=targets.ID.values[ti],item=targets.Item.values[ti],row=targets['row'].values[ti],
                         ax=tx[ti],ay=ty[ti],pex=px[j],pey=py[j],err=m,detected=1))
    else:
        rows.append(dict(ID=targets.ID.values[ti],item=targets.Item.values[ti],row=targets['row'].values[ti],
                         ax=tx[ti],ay=ty[ti],pex=px[j],pey=py[j],err=m,detected=0))
res=pd.DataFrame(rows)
res.to_csv(os.path.join(OUT,'matched_results.csv'), index=False)
np.save(os.path.join(TEMP,'match_res.npy'),res.values)
det=res[res.detected==1]
ND=len(res); NDET=int(res.detected.sum())
print('NDET',NDET,'/ total',ND,' = %.1f%%'%(100*NDET/ND))
print('median err %.2f m'%det.err.median())
print('mean   err %.2f m'%det.err.mean())
print('P50/P90 err: %.2f / %.2f m'%(det.err.median(), det.err.quantile(0.9)))

# ---------- confidence index per detected UXO ----------
flights = df.flight.unique()
def conf4peak(px,py,r=0.6,thr=25):
    npas=0; best=0.0
    for fl in flights:
        g=df[df.flight==fl]
        dd=np.sqrt((g.x.values-px)**2+(g.y.values-py)**2)
        sel=g.anomaly.values[dd<r]
        if len(sel)==0: continue
        m=np.abs(sel).max(); best=max(best,m)
        if m>thr: npas+=1
    cs=100*(1-np.exp(-best/100.0))      # strength component
    cp=100*npas/len(flights)            # multi-flight support component
    return best, npas, 0.5*cs+0.5*cp
res['peak_nT']=0.0; res['n_pass']=0; res['conf']=0.0
for i,t in res.iterrows():
    if t.detected:
        b,np_,c=conf4peak(t.pex,t.pey)
        res.at[i,'peak_nT']=b; res.at[i,'n_pass']=np_; res.at[i,'conf']=c
res.to_csv(os.path.join(OUT,'uxo_detections_conf.csv'), index=False)

# ---------- FIG 1: flight coverage / TMI ----------
fig,ax=plt.subplots(figsize=(7,6),dpi=110)
# tight color window around the regional field so local anomalies are visible
tlo = np.percentile(df.TMI.values, 0.5)
thi = np.percentile(df.TMI.values, 99.5)
sc=ax.scatter(df.x, df.y, c=df.TMI, s=1, cmap='jet', vmin=tlo, vmax=thi, alpha=0.6)
plt.colorbar(sc,ax=ax,label='TMI (nT)')
ax.set_title('Magnetometer survey coverage (TMI)\ncolor scale = %.0f to %.0f nT' % (tlo,thi))
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig1_survey_tmi.png')); plt.close()

# ---------- FIG 2: anomaly map ----------
fig,ax=plt.subplots(figsize=(7,6),dpi=110)
# symmetric color scale clipped to the bulk of actual anomaly readings
vmax2 = np.percentile(np.abs(anom[anom!=0]), 98) if (anom!=0).any() else np.abs(anom).max()
im=ax.imshow(np.flipud(anom), extent=[gx.min(),gx.max(),gy.min(),gy.max()], cmap='RdBu_r', vmin=-vmax2, vmax=vmax2, aspect='equal')
plt.colorbar(im,ax=ax,label='Magnetic anomaly (nT)')
ax.set_title('Gridded magnetic anomaly\ncolor scale = +/-%.0f nT (98th percentile of readings)' % vmax2)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig2_anomaly_map.png')); plt.close()

# ---------- FIG 3: detected peaks on |anomaly| ----------
fig,ax=plt.subplots(figsize=(7,6),dpi=110)
# clip color scale to the bulk of actual readings so contrast is visible
vmax3 = np.percentile(amap[amap>0], 98) if (amap>0).any() else amap.max()
amap_disp = np.where(amap>0, amap, np.nan)   # no-data -> NaN (rendered white)
cm3 = plt.cm.plasma.copy(); cm3.set_bad('white')
im=ax.imshow(np.flipud(amap_disp), extent=[gx.min(),gx.max(),gy.min(),gy.max()], cmap=cm3, vmin=0, vmax=vmax3, aspect='equal')
plt.colorbar(im,ax=ax,label='|anomaly| (nT)')
ax.scatter(peaks_xy[:,0],peaks_xy[:,1],marker='+',s=90,c='cyan',linewidths=1.5,label='detected peaks')
ax.set_title('Detected magnetic anomaly peaks (estimated UXO positions)\ncolor scale = 0 to %.0f nT (98th percentile of readings)' % vmax3)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(loc='upper right')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig3_detected_peaks.png')); plt.close()

# ---------- FIG 4: comparison estimated vs actual (consistent markers with fig7) ----------
fig,ax=plt.subplots(figsize=(7,6),dpi=110)
im=ax.imshow(np.flipud(amap), extent=[gx.min(),gx.max(),gy.min(),gy.max()], cmap='Greys', aspect='equal')
det_r = res[res.detected==1]
miss_r= res[res.detected==0]
sc=ax.scatter(det_r.pex, det_r.pey, c=det_r.conf, cmap='RdYlGn', vmin=0, vmax=100, s=75, edgecolor='k', linewidths=0.4, label='detected (confidence)')
ax.scatter(det_r.ax, det_r.ay, marker='o', s=28, c='none', edgecolor='blue', linewidths=0.8, label='actual (detected)')
for t in det_r.itertuples():
    ax.plot([t.ax,t.pex],[t.ay,t.pey],'-',c='black',lw=0.7,alpha=0.7)
ax.scatter(miss_r.ax, miss_r.ay, marker='x', s=70, c='black', linewidths=1.4, label='missed (not detected)')
plt.colorbar(sc,ax=ax,label='confidence (%)')
# crop to the UXO target band (rows A-E) so row E is at the bottom and row A at the top
ax.set_xlim(targets.x.min()-0.5, targets.x.max()+0.5)
ax.set_ylim(targets.y.min()-0.7, targets.y.max()+0.7)
ax.set_title('Detected ferrous UXO by confidence (estimated vs actual, rows A-E)')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(loc='upper right',fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig4_comparison.png')); plt.close()

# ---------- FIG 5: error histogram ----------
fig,ax=plt.subplots(figsize=(7,5),dpi=110)
ax.hist(det.err, bins=np.arange(0,2.05,0.1), edgecolor='k', alpha=0.8)
ax.axvline(det.err.median(), color='r', ls='--', label='median=%.2f m'%det.err.median())
ax.set_title('Positional error of detected UXO (est. vs actual)')
ax.set_xlabel('horizontal error (m)'); ax.set_ylabel('count'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig5_error_hist.png')); plt.close()

# ---------- FIG 6: per-row detection ----------
fig,ax=plt.subplots(figsize=(7,5),dpi=110)
rows_sum = res.groupby('row').apply(lambda g: pd.Series({'n':len(g),'det':g.detected.sum()})).reset_index()
rows_sum['rate']=100*rows_sum.det/rows_sum.n
ax.bar(rows_sum.row, rows_sum.rate, color='steelblue')
ax.set_title('Detection rate by row')
ax.set_xlabel('row (A=north ... E=south)'); ax.set_ylabel('detection rate (%)')
for i,r in rows_sum.iterrows(): ax.text(i,r.rate+1,'%d/%d'%(r.det,r.n),ha='center')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig6_detection_by_row.png')); plt.close()

# ---------- FIG 7: ALL ferrous UXO targets (identical markers to fig4, full extent) ----------
fig,ax=plt.subplots(figsize=(7,6),dpi=110)
im=ax.imshow(np.flipud(amap), extent=[gx.min(),gx.max(),gy.min(),gy.max()], cmap='Greys', aspect='equal')
det_r = res[res.detected==1]
miss_r= res[res.detected==0]
sc=ax.scatter(det_r.pex, det_r.pey, c=det_r.conf, cmap='RdYlGn', vmin=0, vmax=100, s=75, edgecolor='k', linewidths=0.4, label='detected (confidence)')
ax.scatter(det_r.ax, det_r.ay, marker='o', s=28, c='none', edgecolor='blue', linewidths=0.8, label='actual (detected)')
for t in det_r.itertuples():
    ax.plot([t.ax,t.pex],[t.ay,t.pey],'-',c='black',lw=0.7,alpha=0.7)
ax.scatter(miss_r.ax, miss_r.ay, marker='x', s=70, c='black', linewidths=1.4, label='missed (not detected)')
plt.colorbar(sc,ax=ax,label='confidence (%)')
ax.set_title('All ferrous UXO targets - detected vs missed')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(loc='upper right',fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig7_confidence_map.png')); plt.close()

# ---------- FIG 8: ALL objects (ferrous + non-ferrous) ----------
fig,ax=plt.subplots(figsize=(8,7),dpi=110)
im=ax.imshow(np.flipud(amap_disp), extent=[gx.min(),gx.max(),gy.min(),gy.max()], cmap=cm3, vmin=0, vmax=vmax3, aspect='equal')
# actual objects split by row-target vs clutter and by ferrous/non-ferrous
det_ids = set(res[res.detected==1].ID)                       # detected ferrous A-E
fer_ids  = set(act[act.ferrous].ID)                          # all ferrous
# markers per group
grp=[]
for t in act.itertuples():
    grp.append(dict(x=t.x,y=t.y,fer=t.ferrous,det=t.ID in det_ids,clutter=t.row=='F',id=t.ID))
grp=pd.DataFrame(grp)
def plotgrp(mask,marker,fc,ec,ms,mew,label,alpha=1.0):
    g=grp[mask]
    if len(g): ax.scatter(g.x,g.y,marker=marker,s=ms,c=fc,edgecolors=ec,linewidths=mew,alpha=alpha,label=label)
# ferrous detected (A-E): filled green
plotgrp(grp.fer & grp.det, 'o','lime','black',70,1.0,'ferrous UXO detected')
# ferrous missed (A-E + F): open green ring
plotgrp(grp.fer & (~grp.det) & (~grp.clutter), 'o','none','green',70,1.4,'ferrous missed')
# non-ferrous UXO (A-E): open grey
plotgrp((~grp.fer) & (~grp.clutter), 'o','none','grey',55,1.0,'non-ferrous UXO')
# ferrous clutter row F: filled orange
plotgrp(grp.fer & grp.clutter, 'o','orange','k',55,0.8,'ferrous clutter (F)')
# non-ferrous clutter row F: open grey diamond
plotgrp((~grp.fer) & grp.clutter, 'D','none','grey',45,1.0,'non-ferrous clutter (F)')
# detected peaks
ax.scatter(peaks_xy[:,0],peaks_xy[:,1],marker='+',s=110,c='red',linewidths=1.8,label='detected anomaly peak')
# short pairing lines: actual <-> estimated for detected ferrous UXO
for t in res[res.detected==1].itertuples():
    ax.plot([t.ax,t.pex],[t.ay,t.pey],'-',c='black',lw=0.7,alpha=0.7)
ax.set_title('All objects (ferrous + non-ferrous) vs detected anomaly peaks')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(loc='upper right',fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig8_all_objects.png')); plt.close()

# ---------- save .mat ----------
from scipy.io import savemat
savemat(os.path.join(OUT,'uxo_detection_results.mat'), {
  'survey_x': df.x.values, 'survey_y': df.y.values, 'survey_tmi': df.TMI.values,
  'anomaly_nT': df.anomaly.values, 'anomaly_grid': anom, 'grid_x': gx, 'grid_y': gy,
  'detected_peaks_x': peaks_xy[:,0], 'detected_peaks_y': peaks_xy[:,1], 'peak_magnitude_nT': peaks_xy[:,2],
  'actual_ferrous_x': targets.x.values, 'actual_ferrous_y': targets.y.values,
  'actual_ferrous_ID': targets.ID.values,
  'match_ID': res.ID.values, 'match_err_m': res.err.values, 'match_detected': res.detected.values,
  'match_conf_pct': res.conf.values, 'match_peak_nT': res.peak_nT.values, 'match_n_pass': res.n_pass.values,
  'n_targets': ND, 'n_detected': NDET, 'median_err_m': det.err.median(),
})
print('saved .mat and figures to', OUT)
print('done')