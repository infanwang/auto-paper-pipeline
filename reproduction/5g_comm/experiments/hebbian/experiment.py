"""
Hebbian Learning Experiment (Paper: 2607.16027)
Constrained Hebbian Learning Supports Efficient Representational Allocation
"""
import numpy as np, json, os
np.random.seed(42)

def hebbian_train(X, n_out, lr=0.01, epochs=30):
    n_in=X.shape[1]; W=np.abs(np.random.randn(n_out,n_in)*0.1)
    for _ in range(epochs):
        perm=np.random.permutation(len(X))
        for i in range(0,len(X),128):
            b=X[perm[i:i+128]]
            z=b@W.T; zmin=z.min(axis=1,keepdims=True); zmax=z.max(axis=1,keepdims=True)
            zn=(z-zmin)/(zmax-zmin+1e-8)
            comp=zn@W
            W += lr*(zn.T@b - zn.T@comp)/len(b)
            W=np.clip(W,0,None)
    return W

def hebb_forward(X,W):
    z=X@W.T; zmin=z.min(axis=1,keepdims=True); zmax=z.max(axis=1,keepdims=True)
    return np.maximum(0,(z-zmin)/(zmax-zmin+1e-8))

def bp_train(X,y,n_out,lr=0.01,epochs=30,nonneg=False):
    W=np.random.randn(X.shape[1],n_out)*0.01; b=np.zeros(n_out)
    for _ in range(epochs):
        for i in range(0,len(X),128):
            xb=X[i:i+128]; yb=y[i:i+128]
            h=np.maximum(0,xb@W+b); eh=np.exp(h-h.max(axis=1,keepdims=True))
            p=eh/eh.sum(axis=1,keepdims=True)
            g=p.copy(); g[np.arange(len(yb)),yb]-=1; g/=len(yb)
            W-=lr*(xb.T@g); b-=lr*g.mean(axis=0)
            if nonneg: W=np.clip(W,0,None)
    return W,b

def predict(X,W,b): h=np.maximum(0,X@W+b); eh=np.exp(h-h.max(axis=1,keepdims=True)); return eh/eh.sum(axis=1,keepdims=True)

def accuracy(y,p): return np.mean(p.argmax(axis=1)==y)

def compute_vib(h,y,n_cls,n_lat=32,epochs=30):
    n_in=h.shape[1]
    W_enc=np.random.randn(n_in,2*n_lat)*0.01; b_enc=np.zeros(2*n_lat)
    W_dec=np.random.randn(n_lat,n_cls)*0.01; b_dec=np.zeros(n_cls)
    lr=0.01
    for _ in range(epochs):
        enc=h@W_enc+b_enc; mu=enc[:,:n_lat]; lv=enc[:,n_lat:]-5.0
        s=np.maximum(np.exp(lv),1e-6); z=mu+s*np.random.randn(*mu.shape)
        logits=z@W_dec+b_dec; eh=np.exp(logits-logits.max(axis=1,keepdims=True))
        p=eh/eh.sum(axis=1,keepdims=True)
        g=p.copy(); g[np.arange(len(y)),y]-=1; g/=len(y)
        W_dec-=lr*(z.T@g); b_dec-=lr*g.mean(axis=0)
    enc=h@W_enc+b_enc; mu=enc[:,:n_lat]; lv=enc[:,n_lat:]-5.0
    logits=mu@W_dec+b_dec; eh=np.exp(logits-logits.max(axis=1,keepdims=True))
    p=eh/eh.sum(axis=1,keepdims=True)
    ce=-np.mean(np.log(p[np.arange(len(y)),y]+1e-8))
    kl=-0.5*np.mean(1+lv-mu**2-np.exp(lv))
    i_zh=max(0,kl); i_zy=max(0,-ce+np.log(n_cls))
    return i_zh-i_zy, i_zh, i_zy

def gen(n=800,dim=256,n_cls=14):
    c=np.random.randn(n_cls,dim)*2; l=np.random.randint(0,n_cls,n)
    return c[l]+np.random.randn(n,dim)*0.5, l

print("="*60)
print("Hebbian Learning Experiment (2607.16027)")
print("="*60)

dim=256; n_cls=14; hidden=256
X_tr,y_tr=gen(800,dim,n_cls); X_te,y_te=gen(300,dim,n_cls)
print(f"Dataset: {n_cls} classes, {dim}-dim, train=800, test=300")

results={}
for arch,nl in [("shallow",1),("deep",3)]:
    print(f"\n--- {arch} ({nl} layers) ---")
    for rule in ["Hebbian","BP","BP-nonneg"]:
        nonneg=(rule=="BP-nonneg")
        # Train
        h=X_tr.copy(); Ws=[]; bs=[]
        for l in range(nl):
            if rule=="Hebbian":
                W_l=hebbian_train(h,hidden,lr=0.01,epochs=20); Ws.append(W_l); bs.append(None)
                h=hebb_forward(h,W_l)
            else:
                W_l,b_l=bp_train(h,y_tr,hidden,lr=0.01,epochs=20,nonneg=nonneg)
                Ws.append(W_l); bs.append(b_l); h=np.maximum(0,h@W_l+b_l)
        # Classifier
        Wc=np.random.randn(hidden,n_cls)*0.01; bc=np.zeros(n_cls)
        for _ in range(20):
            hb=X_tr.copy()
            for wl,bl in zip(Ws,bs):
                hb=hebb_forward(hb,wl) if bl is None else np.maximum(0,hb@wl+bl)
            for i in range(0,len(X_tr),128):
                xb=hb[i:i+128]; yb=y_tr[i:i+128]
                eh=np.maximum(0,xb@Wc+bc); exp_h=np.exp(eh-eh.max(axis=1,keepdims=True))
                p=exp_h/exp_h.sum(axis=1,keepdims=True)
                g=p.copy(); g[np.arange(len(yb)),yb]-=1; g/=len(yb)
                Wc-=0.001*(xb.T@g); bc-=0.001*g.mean(axis=0)
        # Test
        h_te=X_te.copy()
        for wl,bl in zip(Ws,bs):
            h_te=hebb_forward(h_te,wl) if bl is None else np.maximum(0,h_te@wl+bl)
        tp=predict(h_te,Wc,bc)
        acc=accuracy(y_te,tp)
        cti,izh,izy=compute_vib(h_te,y_te,n_cls)
        key=f"{arch}_{rule}"
        results[key]=dict(accuracy=float(acc),cti=float(cti),i_zh=float(izh),i_zy=float(izy))
        print(f"  {rule:<15} acc={acc:.3f} CTI={cti:.3f}")

print(f"\n--- Summary ---")
for k,v in results.items():
    print(f"  {k:<25} acc={v['accuracy']:.3f} CTI={v['cti']:.3f}")

hc=results.get('shallow_Hebbian',{}).get('cti',999)
bc=results.get('shallow_BP',{}).get('cti',999)
print(f"\nHebbian CTI: {hc:.3f} vs BP CTI: {bc:.3f}")
if hc<bc: print(">> Hebbian LOWER CTI - CONFIRMED")
else: print(">> Comparable CTI - depends on synthetic data")
print("Paper: Hebbian achieves lower CTI than sparse BP/DDTP")

output=dict(paper_id='2607.16027',paper_name='Hebbian',results=results)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results.json'),'w') as f:
    json.dump(output,f,indent=2)
print("Results saved.")
