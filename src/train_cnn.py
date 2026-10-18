"""Optional CPU CNN training. Requires annotated segment rows and PyTorch."""
from pathlib import Path
import json,random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from .preprocessing import log_mel
from .segmentation import read_segment
class LungCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2))
    def forward(self,x): return self.net(x)
def train_cnn(rows,output_path,epochs=25,batch_size=16,lr=.001,seed=42,target_sr=22050):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed); X=[];Y=[];groups=[]
    for r in rows:
        a,sr=read_segment(r,target_sr); s=log_mel(a,sr); s=np.pad(s,((0,0),(0,max(0,256-s.shape[1]))))[:,:256];X.append(s[None]);Y.append([r['crackles'],r['wheezes']]);groups.append(str(r.get('patient_id') or r['recording_id']))
    groups=np.asarray(groups); unique=np.unique(groups)
    if len(unique)<2: raise ValueError('Need at least two patient/recording groups for validation split')
    rng=np.random.default_rng(seed);rng.shuffle(unique); val=set(unique[:max(1,int(.2*len(unique)))]); vi=np.array([g in val for g in groups]);ti=~vi
    if not ti.any(): raise ValueError('Not enough groups for split')
    X=torch.tensor(np.asarray(X),dtype=torch.float32);Y=torch.tensor(np.asarray(Y),dtype=torch.float32);train=DataLoader(TensorDataset(X[ti],Y[ti]),batch_size=batch_size,shuffle=True);valid=DataLoader(TensorDataset(X[vi],Y[vi]),batch_size=batch_size)
    model=LungCNN();opt=torch.optim.Adam(model.parameters(),lr=lr);lossfn=nn.BCEWithLogitsLoss();hist={'train_loss':[],'val_loss':[]};best=float('inf');stale=0
    for ep in range(epochs):
        model.train();ls=[]
        for xb,yb in train: opt.zero_grad();loss=lossfn(model(xb),yb);loss.backward();opt.step();ls.append(loss.item())
        model.eval();vs=[]
        with torch.no_grad():
            for xb,yb in valid:vs.append(lossfn(model(xb),yb).item())
        tr=float(np.mean(ls));va=float(np.mean(vs));hist['train_loss'].append(tr);hist['val_loss'].append(va)
        if va<best:
            best=va;stale=0;Path(output_path).parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'threshold':.5,'seed':seed},output_path)
        else:stale+=1
        if stale>=5:break
    Path(output_path).with_suffix('.history.json').write_text(json.dumps(hist,indent=2));return hist
