"""Generate source-backed tables, geographic figures, and bounded Ridge results."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyBboxPatch,FancyArrowPatch
from threadpoolctl import threadpool_limits
import prepare_revision as p

HERE=p.HERE;REV=p.REV
OUT=HERE/'generated';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})
TEAL='#27647b';ORANGE='#c76d30';PURPLE='#7a5290'


def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)


def land(ax):
    geo=json.loads((HERE/'ne_110m_land.geojson').read_text())
    for f in geo['features']:
        g=f['geometry'];polys=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
        for poly in polys:
            ax.add_patch(Polygon(np.asarray(poly[0]),facecolor='#e5e6e3',edgecolor='#b3b6b3',lw=.35,zorder=0))


def geography():
    raw=p.LIB.parse_hurdat2(str(p.SOURCE))
    manifest=json.loads((REV/'followup/window_selection/window_manifest.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(6.5,7.4),sharex=True,sharey=True,layout='constrained')
    info={}
    for ax,period,title in zip(axes,('test2022_2024','extension2025'),('2022–2024','2025 extension')):
        rows=[r for r in manifest if r['period']==period]
        ids={r['storm_id'] for r in rows}
        land(ax)
        for sid in sorted(ids):
            g=raw[raw.typhoon_id==sid].sort_values(['date','time'])
            ax.plot(g.longitude,g.latitude,color='#b4c0c6',lw=.7,zorder=1)
        for original,color,marker,label in ((True,TEAL,'o','Original'),(False,ORANGE,'^','Added')):
            selected=[r for r in rows if r['accepted_original']==original]
            coords=[]
            for r in selected:
                q=raw[(raw.typhoon_id==r['storm_id'])&(raw._timestamp_hours==r['initialization_epoch_hour'])&raw.time.isin(p.LIB.STANDARD_TIMES)]
                assert len(q)==1
                coords.append(q[['longitude','latitude']].iloc[0].to_numpy())
            xy=np.asarray(coords)
            count=len({r['storm_id'] for r in selected})
            ax.scatter(xy[:,0],xy[:,1],s=9,marker=marker,c=color,alpha=.65,
                label=f'{label}: {len(selected)} windows, {count} '+('storm' if count==1 else 'storms'),zorder=3)
        events=raw[(raw.typhoon_id.isin(ids))&(raw.record_identifier=='L')]
        ax.scatter(events.longitude,events.latitude,s=19,marker='x',c='#9a3154',lw=.85,label='Recorded L positions',zorder=4)
        ax.set(xlim=(-105,-10),ylim=(0,65),xlabel='Longitude (degrees east)',title=title)
        ax.grid(alpha=.2,lw=.5);ax.legend(loc='upper right',fontsize=9,framealpha=.95)
        info[period]=dict(windows=len(rows),storms=len(ids),L_records=len(events))
    axes[0].set_ylabel('Latitude (degrees north)')
    save(fig,'coverage_map');p.dump(OUT/'geography_counts.json',info)


def cases():
    root=REV/'followup/window_selection/results/synoptic_features/test2022_2024'
    agg=np.load(root/'aggregate_errors.npz')
    ids=agg['storm_id'];win=agg['window']
    single=np.load(root/'lead4_residual_rmw_on_seed42.npz')['pred_error_km'].reshape(-1)
    direct=np.load(root/'lead4_direct_rmw_on_seed42.npz')['pred_error_km'].reshape(-1)
    delta=single-direct
    means={s:float(delta[ids==s].mean()) for s in np.unique(ids)}
    choices=[]
    for sign in (-1,1):
        candidates=sorted([s for s,v in means.items() if sign*v>0],key=lambda s:(means[s],s))
        assert candidates
        sid=candidates[(len(candidates)-1)//2]
        ix=np.flatnonzero((ids==sid)&(sign*delta>0));order=ix[np.argsort(delta[ix],kind='stable')]
        choices.append(int(order[(len(order)-1)//2]))
    raw=p.LIB.parse_hurdat2(str(p.SOURCE));fig,axes=plt.subplots(2,1,figsize=(6.5,7.0),layout='constrained')
    selected=[]
    for ax,ix,label in zip(axes,choices,('Favorable storm stratum','Unfavorable storm stratum')):
        sid=str(ids[ix]);w=int(win[ix]);npzs={name:np.load(root/f'{stem}.npz') for name,stem in
            [('CV','CV'),('Ridge','Ridge_on'),('Direct','lead4_direct_rmw_on_seed42'),('Single residual','lead4_residual_rmw_on_seed42')]}
        row=np.load(REV/'followup/window_selection/data/synoptic_features/test2022_2024/test'/f'{sid}_{w:05d}.npy',allow_pickle=True).item()
        hist=row['history_latlon'];init=int(row['history_timestamp_hours'][-1]);
        g=raw[(raw.typhoon_id==sid)&raw.time.isin(p.LIB.STANDARD_TIMES)&raw._timestamp_hours.between(init,init+24)]
        land(ax);ax.plot(hist[:,1],hist[:,0],color='#6e777c',lw=1,label='Observed history')
        ax.plot(g.longitude,g.latitude,'o-',color='black',ms=3,lw=1.4,label='Observed future')
        ax.scatter(hist[-1,1],hist[-1,0],c='black',marker='s',s=35,label='Initialization',zorder=5)
        coords=[hist,g[['latitude','longitude']].to_numpy()]
        errors={}
        for (name,data),color,marker in zip(npzs.items(),('#4a7962',PURPLE,ORANGE,TEAL),('^','D','x','P')):
            key='pred_latlon' if 'pred_latlon' in data else 'pred_latlon'
            jj=np.flatnonzero((data['storm_id']==sid)&(data['window']==w));assert len(jj)==1
            pos=data[key][jj[0]].reshape(-1,2)[-1];coords.append(pos[None])
            ax.scatter(pos[1],pos[0],s=56,c=color,marker=marker,label=name,zorder=6)
            ekey='error_km' if 'error_km' in data else 'pred_error_km'
            errors[name]=float(data[ekey][jj[0]].reshape(-1)[-1])
        xy=np.concatenate(coords);pad=1.5
        ax.set(xlim=(xy[:,1].min()-pad,xy[:,1].max()+pad),ylim=(xy[:,0].min()-pad,xy[:,0].max()+pad),
            title=f"{sid.split('_',1)[1]} | {np.datetime64(init,'h')} UTC\n{label}",xlabel='Longitude (degrees east)')
        ax.grid(alpha=.2)
        handles, labels=ax.get_legend_handles_labels()
        labels=[f'{label}: {errors[label]:.1f} km' if label in errors else label for label in labels]
        ax.legend(handles,labels,fontsize=9,loc='best')
        selected.append(dict(storm_id=sid,window=w,initialization_epoch_hour=init,
            selection_storm_delta_km=means[sid],selection_window_delta_km=float(delta[ix]),display_seed=42,endpoint_errors_km=errors))
    axes[0].set_ylabel('Latitude (degrees north)');save(fig,'forecast_cases')
    p.dump(OUT/'case_selection.json',dict(policy='For fixed display seed 42, partition storms by sign of storm-mean single-residual-minus-direct error. Select lower median storm in each stratum; select lower median window among sign-consistent windows in that storm. RMW on, frozen synoptic-feature policy; only 24 h endpoints. Descriptive retrospective selection, not evidence of prevalence.',cases=selected))


def diagrams():
    fig,axes=plt.subplots(2,1,figsize=(6.5,4.5),layout='constrained',height_ratios=[1,1.2])
    ax=axes[0];ax.axis('off');ax.set(xlim=(0,10),ylim=(0,2))
    texts=[('Historical input',.2,1.0),('Same encoder\narchitecture',2.4,1.0),('Separate output heads',4.6,1.0)]
    for text,x,y in texts:
        ax.text(x,y,text,va='center',bbox=dict(boxstyle='round,pad=.5',fc='#eef4f6',ec=TEAL))
    for x in (1.65,3.85):ax.annotate('',xy=(x+.55,1),xytext=(x,1),arrowprops=dict(arrowstyle='->',color=TEAL))
    ax.text(7.2,1.05,'A: direct displacement\nB: CV + residual\nC: fixed 0.5 mixture\nD: learned mixture',va='center',fontsize=9)
    ax.set_title('(a) Matched forecasting comparison',loc='left',fontsize=10,fontweight='bold')
    ax=axes[1];ax.set_xlim(-70,28);ax.set_ylim(-.9,2);ax.set_yticks([])
    ts=np.arange(-66,25,6)
    ax.plot(ts,np.ones(len(ts)),color=TEAL,lw=1)
    ax.scatter(ts,np.ones(len(ts)),c=TEAL,s=22,zorder=3)
    ax.scatter([9],[1],c=ORANGE,marker='D',s=45,zorder=4)
    ax.axvline(0,color='#444',ls='--',lw=.8);ax.axvspan(0,24,color='#f9eddf',alpha=.6)
    ax.annotate('Additional L record at T + 9 h',xy=(9,1),xytext=(-30,1.7),arrowprops=dict(arrowstyle='->'),fontsize=9)
    ax.text(-66,.35,'12 historical positions: T − 66 h to T',fontsize=9)
    ax.text(2,.35,'Future: (T, T + 24 h]',fontsize=9)
    ax.text(-66,-.25,'Raw-block rule: rejects the interrupted 16-row candidate',color=ORANGE,fontsize=9)
    ax.text(-66,-.65,'Synoptic-first: retains required timestamps; L annotates the future subset',color=TEAL,fontsize=9)
    ax.set_xticks([-66,0,6,12,18,24],['−66','T','+6','+12','+18','+24']);ax.set_xlabel('Hours relative to initialization')
    ax.set_title('(b) Schematic future event and window eligibility',loc='left',fontsize=10,fontweight='bold')
    save(fig,'prediction_and_timeline')
    fig,ax=plt.subplots(figsize=(6.5,5.5));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,5.5))
    boxes=[(.15,4.0,'Versioned raw records\nstorm ID +\nexact timestamps'),(5.25,4.0,'Select synoptic times\nrequire 12 + 4\ncomplete records'),
           (.15,2.35,'Raw event records\nL in (T, T + 24 h]'),(5.25,2.35,'Historical features\nfit normalization\non train only'),
           (.15,.7,'Traceable manifest\nsource hash + policy\n+ timestamps'),(5.25,.7,'Matched forecasts/errors\npaired whole-storm\nresampling')]
    for x,y,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),4.25,1.1,boxstyle='round,pad=.08',fc='#eef4f6',ec=TEAL,lw=1))
        ax.text(x+2.125,y+.55,text,ha='center',va='center',fontsize=12)
    for a,b in [((4.5,4.55),(5.1,4.55)),((2.275,3.85),(2.275,3.6)),((7.375,3.85),(7.375,3.6)),
                ((2.275,2.2),(2.275,1.95)),((7.375,2.2),(7.375,1.95)),((4.5,1.25),(5.1,1.25))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=TEAL))
    ax.plot([5.1,4.83,4.83],[4.35,4.35,1.55],color=TEAL,lw=1)
    ax.add_patch(FancyArrowPatch((4.83,1.55),(4.48,1.55),arrowstyle='-|>',mutation_scale=12,color=TEAL))
    ax.text(5,.2,'Executable checks and score reconstruction',ha='center',fontsize=12)
    fig.tight_layout();save(fig,'auditable_workflow')


def ridge():
    sys.path.insert(0,str(REV/'experiments'))
    import ridge_baseline as rb
    rb.ALPHAS=np.asarray(json.loads((HERE/'design_lock.json').read_text())['designs']['ridge_alphas'])
    lib=rb.reference_functions();meta=json.loads((HERE/'data/main/metadata.json').read_text())
    keep=[i for i,n in enumerate(meta['feature_names']) if n!='rmax']
    with threadpool_limits(limits=4):
        train=lib.load_records(HERE/'data/main/train',[4]);val=lib.load_records(HERE/'data/main/val',[4])
        for data in (train,val):
            for key in ('x','y_raw','cv_raw'):data[key]=data[key].astype(np.float64)
        model=rb.fit_design(train,val,keep,40,lib,True)
        np.savez_compressed(HERE/'ridge_model.npz',**{k:model[k] for k in ('coef','intercept','x_mean','x_std','keep')},alpha=model['alpha'])
        p.dump(HERE/'ridge_selection.json',dict(alpha=model['alpha'],tuning=model['tuning'],equivalence_check=model['equivalence_check']))
        for period,folder in [('test','main'),('extension','extension')]:
            data=lib.load_records(HERE/f'data/{folder}/test',[4])
            rb.save_predictions(HERE/f'ridge_{period}_predictions.npz',rb.predict(model,data,40),data,lib)
            rb.save_predictions(HERE/f'cv_{period}_predictions.npz',data['cv_raw'],data,lib)
    print('Ridge fitted; validation-selected alpha =',model['alpha'],flush=True)


def temporal_applicability():
    raw=p.LIB.parse_hurdat2(str(p.SOURCE));raw=raw[raw.storm_year.between(1980,2003)]
    stats=dict(period='1980-2003',source_sha256=p.sha(p.SOURCE),raw_windows=0,synoptic_windows=0,
        raw_storms=set(),synoptic_storms=set(),future_L_windows=0,future_L_storms=set())
    for sid,g in raw.groupby('typhoon_id'):
        p.validate(g)
        arr=p.storm_arrays(g);indices=p.accepted_indices(arr)
        old=g.sort_values(['date','time']);times=old.time.to_numpy();ts=old._timestamp_hours.to_numpy()
        n=sum(np.all(np.isin(times[j:j+16],list(p.LIB.STANDARD_TIMES))) and np.all(np.diff(ts[j:j+16])==6) for j in range(max(0,len(g)-15)))
        stats['raw_windows']+=int(n);stats['synoptic_windows']+=len(indices)
        if n:stats['raw_storms'].add(sid)
        if indices:stats['synoptic_storms'].add(sid)
        records=p.make_rows(g,{'test':{sid}})
        nf=sum(r['recorded_future_landfall'] for r in records)
        stats['future_L_windows']+=nf
        if nf:stats['future_L_storms'].add(sid)
    for key in ('raw_storms','synoptic_storms','future_L_storms'):stats[key]=len(stats[key])
    stats['scope']='Independent historical-period application in the same Atlantic archive and format; not a second basin/source, not a neural transfer test.'
    p.dump(OUT/'temporal_applicability.json',stats);print(stats,flush=True)


if __name__=='__main__':
    ridge();diagrams();geography();cases();temporal_applicability()
