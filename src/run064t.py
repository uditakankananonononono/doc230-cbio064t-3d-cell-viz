import glob,os,sys,json,time,numpy as np,tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu,gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
D='data/'; OUT='results/cache'; os.makedirs(OUT,exist_ok=True)
def stacks(split,embs=None):
    out=[]
    for f in sorted(glob.glob(f'{D}img/Images/{split}/Images/*.tif')):
        n=os.path.basename(f)
        if embs and not any(n.startswith(e+'_') for e in embs): continue
        out.append((n[:-4],f,f'{D}gt/GroundTruth/{split}/GroundTruth_QCANet/{n}'))
    return out
def emb(n): return int(n.split('_')[0].replace('Emb',''))
def f1_inst(p,g,thr=0.5):
    P=int(p.max())+1;G=int(g.max())+1
    c=np.bincount(p.ravel().astype(np.int64)*G+g.ravel().astype(np.int64),minlength=P*G).reshape(P,G)
    ap=c.sum(1);ag=c.sum(0);iou=c/(ap[:,None]+ag[None,:]-c+1e-9)
    iou[0,:]=0;iou[:,0]=0
    used_p=set();used_g=set();tp=0
    for i,j in sorted(zip(*np.where(iou>=thr)),key=lambda ij:-iou[ij]):
        if i in used_p or j in used_g: continue
        used_p.add(i);used_g.add(j);tp+=1
    npred=len([i for i in range(1,P) if ap[i]>0]);ngt=len([j for j in range(1,G) if ag[j]>0])
    fp=npred-tp;fn=ngt-tp
    return (2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 1.0),npred,ngt
def dice(a,b): s=a.sum()+b.sum(); return 2*(a&b).sum()/s if s>0 else 1.0
def score(p,g):
    p=p.astype(np.int32);g=g.astype(np.int32)
    f1,npred,ngt=f1_inst(p,g); return dict(dice=float(dice(p>0,g>0)),f1=float(f1),npred=npred,ngt=ngt,cnt_err=npred-ngt)
def baseline(a):
    s=gaussian(a.astype(np.float32),1,preserve_range=True); m=s>threshold_otsu(s)
    dt=ndi.distance_transform_edt(m); dts=gaussian(dt,1)
    pk=peak_local_max(dts,min_distance=4,labels=m.astype(int),exclude_border=False)
    mk=np.zeros(a.shape,np.int32); mk[tuple(pk.T)]=np.arange(1,len(pk)+1)
    return watershed(-dts,mk,mask=m)
_model=None
def model():
    global _model
    if _model is None:
        from cellpose import models; _model=models.Cellpose(gpu=False,model_type='nuclei')
    return _model
def cp(a,diam=17,mode='3d'):
    kw=dict(do_3D=True) if mode=='3d' else dict(do_3D=False,stitch_threshold=0.5)
    return model().eval(a.astype(np.float32),diameter=diam,channels=[0,0],z_axis=0,**kw)[0]
def run(name,tag,split,embs,fn):
    path=f'{OUT}/{tag}.json'; res=json.load(open(path)) if os.path.exists(path) else {}
    for n,fi,fg in stacks(split,embs):
        if n in res: continue
        a=tifffile.imread(fi); g=tifffile.imread(fg); t=time.time()
        res[n]=fn(a,g); res[n]['sec']=round(time.time()-t,1)
        json.dump(res,open(path,'w')); print(tag,n,res[n],flush=True)
    return res
def noisy(a):
    r=np.random.RandomState(0); lo,hi=np.percentile(a,[1,99]); return a+r.normal(0,0.10*(hi-lo),a.shape)
def f_cp(diam,mode):
    return lambda a,g: score(cp(a,diam,mode),g)
def f_primary(a,g):
    p=cp(a,17,'3d'); r=score(p,g); r['zflip_dice']=float(dice(p>0,g[::-1]>0))
    q=p.copy(); ids,cnt=np.unique(q[q>0],return_counts=True)
    for i,c in zip(ids,cnt):
        if c<100: q[q==i]=0
    q,_,_=__import__('skimage.segmentation',fromlist=['relabel_sequential']).relabel_sequential(q)
    r['filtered']=score(q,g); return r
def f_zsub(a,g): return score(cp(a[::2],17,'2d'),g[::2])
JOBS=[('primary','primary_test','test',None,f_primary),
 ('baseline','baseline_test','test',None,lambda a,g:score(baseline(a),g))]
for d in (8,12,17,24): JOBS.append((f'stitch_d{d}',f'stitch_d{d}_test','test',None,f_cp(d,'2d')))
JOBS+= [('noise','noise_test','test',None,lambda a,g:score(cp(noisy(a),17,'2d'),g)),
 ('zsub','zsub_test','test',None,f_zsub),
 ('stitch_d17_traindev','stitch_d17_train','train',['Emb01','Emb02','Emb03','Emb04'],f_cp(17,'2d'))]
if __name__=='__main__':
    sel=sys.argv[1:] 
    for name,tag,split,embs,fn in JOBS:
        if sel and name not in sel: continue
        print('JOB',name,flush=True); run(name,tag,split,embs,fn)
    print('DONE',flush=True)
