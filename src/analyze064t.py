# Aggregation only; gates are exactly those in PREREG.md. Reads results/cache/*.json.
import json,numpy as np,re
C='results/cache/'
def L(t): return json.load(open(C+t+'.json'))
rs=np.random.RandomState(0)
def boot(x,B=2000):
    x=np.asarray(x,float); m=[x[rs.randint(0,len(x),len(x))].mean() for _ in range(B)]; return [float(np.percentile(m,2.5)),float(np.percentile(m,97.5))]
def emb(n): return int(n.split('_')[0].replace('Emb',''))
def summ(d,key):
    v=np.array([r[key] for r in d.values()]); return dict(mean=float(v.mean()),median=float(np.median(v)),ci95=boot(v),n=len(v))
out={}
P=L('primary_test'); B=L('baseline_test')
out['D1']=dict(n_test=len(P),ngt_min=min(r['ngt'] for r in P.values()),ngt_max=max(r['ngt'] for r in P.values()),ngt_total=sum(r['ngt'] for r in P.values()))
out['D2']=dict(dice=summ(P,'dice'),f1=summ(P,'f1'),cnt_err_mean=float(np.mean([r['cnt_err'] for r in P.values()])))
out['D2']['gate_agreement']=bool(out['D2']['dice']['median']>=0.80 and out['D2']['f1']['mean']>=0.70)
out['D3']=dict(baseline_dice=summ(B,'dice'),baseline_f1=summ(B,'f1'));out['D3']['diff_f1']=out['D2']['f1']['mean']-out['D3']['baseline_f1']['mean'];out['D3']['gate_adds_value']=bool(out['D3']['diff_f1']>=0.05)
S={d:L(f'stitch_d{d}_test') for d in (8,12,17,24)}
out['D4']={str(d):summ(S[d],'f1') for d in S};rng=max(v['mean'] for v in out['D4'].values())-min(v['mean'] for v in out['D4'].values());out['D4']['range_f1']=rng;out['D4']['gate_diameter_sensitive']=bool(rng>0.10)
out['D5']=dict(f1_3d=out['D2']['f1']['mean'],f1_stitch17=out['D4']['17']['mean']);out['D5']['diff']=out['D5']['f1_3d']-out['D5']['f1_stitch17'];out['D5']['gate_modes_differ']=bool(abs(out['D5']['diff'])>=0.05)
st={'<=4':lambda n:n<=4,'5-16':lambda n:5<=n<=16,'>16':lambda n:n>16};out['D6']={}
for k,f in st.items():
    v=[r['f1'] for r in P.values() if f(r['ngt'])]; out['D6'][k]=dict(n=len(v),f1=float(np.mean(v)) if v else None)
out['D6']['gate_high_density_ok']=bool(out['D6']['>16']['f1'] is not None and out['D6']['>16']['f1']>=0.5)
out['D7']={}
for e in sorted({emb(n) for n in P}):
    out['D7'][f'Emb{e}']=dict(f1=float(np.mean([r['f1'] for n,r in P.items() if emb(n)==e])),dice=float(np.mean([r['dice'] for n,r in P.items() if emb(n)==e])))
f=[v['f1'] for k,v in out['D7'].items()];out['D7']['sd_f1_ddof1']=float(np.std(f,ddof=1))
T=L('stitch_d17_train');out['D8']=dict(train_device_f1=summ(T,'f1'),test_device_f1=out['D4']['17']);out['D8']['diff']=out['D8']['train_device_f1']['mean']-out['D8']['test_device_f1']['mean'];out['D8']['gate_device_dependent']=bool(abs(out['D8']['diff'])>=0.10)
N=L('noise_test');Z=L('zsub_test');b=out['D4']['17']['mean']
out['D9']=dict(noise_f1=summ(N,'f1'),zsub_f1=summ(Z,'f1'),clean_stitch_f1=b);out['D9']['drop_noise']=b-out['D9']['noise_f1']['mean'];out['D9']['drop_zsub']=b-out['D9']['zsub_f1']['mean'];out['D9']['gate_fragile']=bool(out['D9']['drop_noise']>=0.10 or out['D9']['drop_zsub']>=0.10)
zf=float(np.mean([r['zflip_dice'] for r in P.values()]));out['D9']['zflip_dice_mean']=zf;out['D9']['gate_metric_nontrivial']=bool(out['D2']['dice']['mean']-zf>=0.20)
out['D10']=dict(f1_before=out['D2']['f1']['mean'],f1_after=float(np.mean([r['filtered']['f1'] for r in P.values()])),cnt_err_before=out['D2']['cnt_err_mean'],cnt_err_after=float(np.mean([r['filtered']['cnt_err'] for r in P.values()])))
json.dump(out,open('results/results064t.json','w'),indent=1);print(json.dumps(out,indent=1))
