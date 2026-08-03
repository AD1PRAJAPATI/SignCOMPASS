"""Fine-tune SHuBERT end-to-end for ISLR on ASL-Citizen.
Feeds the 4 input streams (face/lh/rh DINOv2 384-d, body 14-d) into SHuBERT,
pools the encoder output, ArcFace head. Goal: SHuBERT at its real ~68%.
Run from $ISLR with PYTHONPATH including $REPO/fairseq (the sbatch sets it)."""
import argparse,os,random,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch.amp import GradScaler,autocast
from torch.optim import AdamW
from torch.utils.data import Dataset,DataLoader
from dataset_islr import GlossVocabISLR,load_metadata,get_splits,make_class_balanced_sampler
from train_islr import topk_accuracy,cosine_warmup_schedule
from models.fusion_model import AttentionPool,ArcFaceHead

SUF={"face":"_face","lh":"_hand1","rh":"_hand2","body":"_pose"}

class FTData(Dataset):
    def __init__(self,df,vocab,dirs,mf):
        self.vc=vocab;self.dirs=dirs;self.mf=mf;df=df.reset_index(drop=True)
        keep=[i for i,(_,r) in enumerate(df.iterrows()) if self._ok(r["video_id"])]
        if len(keep)<len(df):print(f"[FTData] dropped {len(df)-len(keep)}/{len(df)} missing")
        self.df=df.iloc[keep].reset_index(drop=True)
    def _p(self,s,v):return os.path.join(self.dirs[s],v+SUF[s]+".npy")
    def _ok(self,v):return all(os.path.exists(self._p(s,v)) for s in SUF)
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];v=r["video_id"];a={s:np.load(self._p(s,v)) for s in SUF}
        T=min(min(len(a[s]) for s in SUF),self.mf)
        return {**{s:torch.from_numpy(a[s][:T]).float() for s in SUF},
                "len":T,"label":self.vc.encode(r["gloss"])}

def collate(B):
    mT=max(b["len"] for b in B)
    def pad(x):return torch.cat([x,torch.zeros(mT-x.shape[0],x.shape[1])],0) if x.shape[0]<mT else x
    o={s:torch.stack([pad(b[s]) for b in B]) for s in SUF}
    o["lengths"]=torch.tensor([b["len"] for b in B]);o["labels"]=torch.tensor([b["label"] for b in B]);return o

def build(root,dirs,mf,seed):
    asl=os.path.join(root,"data",os.environ.get("ISLR_DATASET","asl_citizen"))
    df=get_splits(load_metadata(os.path.join(asl,"metadata.csv")),seed=seed)
    vc=GlossVocabISLR();vc.build(df[df.split=="train"]["gloss"].tolist())
    df=df[df["gloss"].isin(set(vc._gloss2id))].reset_index(drop=True)
    return (FTData(df[df.split=="train"],vc,dirs,mf),FTData(df[df.split=="val"],vc,dirs,mf),
            FTData(df[df.split=="test"],vc,dirs,mf),vc)

def load_sh(ckpt):
    from examples.shubert.models.shubert import SHubertModel,SHubertConfig
    m=SHubertModel(SHubertConfig());sd=torch.load(ckpt,map_location="cpu");sd=sd.get("model",sd)
    m.load_state_dict(sd,strict=False);return m

class SHFT(nn.Module):
    def __init__(self,ckpt,nc,d=768,freeze=False,dropout=0.3):
        super().__init__();self.sh=load_sh(ckpt)
        if freeze:
            for p in self.sh.parameters():p.requires_grad=False
        self.norm=nn.LayerNorm(d);self.pool=AttentionPool(d,dropout=dropout)
        self.head=ArcFaceHead(d,nc,margin=0.3,scale=64.0)
    def forward(self,b,labels=None):
        dev=next(self.parameters()).device;B,T=b["face"].shape[:2];z=torch.zeros(T,1,device=dev)
        src=[{"face":b["face"][i],"left_hand":b["lh"][i],"right_hand":b["rh"][i],"body_posture":b["body"][i],
              "label_face":z,"label_left_hand":z,"label_right_hand":z,"label_body_posture":z} for i in range(B)]
        ln=b["lengths"].to(dev);pad=torch.arange(T,device=dev)[None,:]>=ln[:,None]
        x=self.sh(src,padding_mask=pad,mask=False,features_only=True)["x"]
        return {"logits":self.head(F.normalize(self.norm(self.pool(x,ln)),dim=-1),labels)}

@torch.no_grad()
def ev(m,dl,dev):
    m.eval();L=[];Y=[]
    for b in dl:
        for s in SUF:b[s]=b[s].to(dev)
        L.append(m(b)["logits"].cpu());Y.append(b["labels"])
    if not L: return 0.0,0.0,0.0
    a=topk_accuracy(torch.cat(L),torch.cat(Y),(1,5,10));return a[1],a[5],a[10]

def main(a):
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device("cuda")
    asl=os.path.join(a.data_root,"data",os.environ.get("ISLR_DATASET","asl_citizen"),"shubert")
    dirs={"face":os.path.join(asl,"face_feats"),"lh":os.path.join(asl,"hand1_feats"),
          "rh":os.path.join(asl,"hand2_feats"),"body":os.path.join(asl,"body_feats")}
    tr,va,te,vc=build(a.data_root,dirs,a.max_frames,a.seed);NC=len(vc)
    print(f"classes={NC} train={len(tr)} val={len(va)} test={len(te)}")
    samp=make_class_balanced_sampler(tr.df,vc)
    dl=lambda ds,s=None:DataLoader(ds,batch_size=a.batch_size,sampler=s,
        shuffle=(s is None and ds is tr),num_workers=a.num_workers,collate_fn=collate,
        pin_memory=True,multiprocessing_context="forkserver")
    trdl,vadl,tedl=dl(tr,samp),dl(va),dl(te)
    m=SHFT(a.ckpt,NC,freeze=a.freeze_backbone,dropout=a.dropout).to(dev)
    if getattr(a, "init_ft", None) and os.path.isfile(a.init_ft):
        sd = torch.load(a.init_ft, map_location="cpu")
        m.load_state_dict(sd.get("model_state", sd), strict=True)
        print(f"[resume] loaded FT weights from {a.init_ft}")
    bb=[p for n,p in m.named_parameters() if n.startswith("sh.") and p.requires_grad]
    hd=[p for n,p in m.named_parameters() if not n.startswith("sh.")]
    groups=[{"params":hd,"lr":a.lr}]+([{"params":bb,"lr":a.backbone_lr}] if bb else [])
    opt=AdamW(groups,weight_decay=a.wd);sched=cosine_warmup_schedule(opt,a.warmup_epochs,a.epochs)
    scaler=GradScaler("cuda");best=0.0;pat=0;os.makedirs(a.save_dir,exist_ok=True)
    for ep in range(1,a.epochs+1):
        m.train();t0=time.time()
        for b in trdl:
            for s in SUF:b[s]=b[s].to(dev)
            y=b["labels"].to(dev);opt.zero_grad()
            with autocast("cuda"):loss=F.cross_entropy(m(b,labels=y)["logits"],y,label_smoothing=0.1)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(),5.0);scaler.step(opt);scaler.update()
        sched.step();v=ev(m,vadl,dev)
        print(f"Ep {ep:3d}/{a.epochs} | val top1={v[0]:.2f} top5={v[1]:.2f} R@10={v[2]:.2f} | {time.time()-t0:.0f}s")
        if v[0]>best:best=v[0];pat=0;torch.save({"model_state":m.state_dict()},os.path.join(a.save_dir,"best.pt"));print(f"  ✓ best {best:.2f}")
        else:
            pat+=1
            if pat>=a.patience:print("early stop");break
    m.load_state_dict(torch.load(os.path.join(a.save_dir,"best.pt"))["model_state"]);t=ev(m,tedl,dev)
    print(f"\nSHUBERT_FT_RESULT: freeze={a.freeze_backbone} test_top1={t[0]:.2f} test_top5={t[1]:.2f} test_r10={t[2]:.2f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",default="/project2/jessetho_1732/aditeya/data/shubert_weights/shubert.pt")
    p.add_argument("--init_ft",default=None,help="resume/warm-start from a prior SHFT best.pt")
    p.add_argument("--data_root",default="/project2/jessetho_1732/aditeya");p.add_argument("--save_dir",default="ckpt_shubert_ft")
    p.add_argument("--freeze_backbone",action="store_true")
    p.add_argument("--epochs",type=int,default=40);p.add_argument("--warmup_epochs",type=int,default=3)
    p.add_argument("--batch_size",type=int,default=32);p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--backbone_lr",type=float,default=1e-5);p.add_argument("--wd",type=float,default=1e-4)
    p.add_argument("--dropout",type=float,default=0.3);p.add_argument("--max_frames",type=int,default=64)
    p.add_argument("--patience",type=int,default=10);p.add_argument("--num_workers",type=int,default=8);p.add_argument("--seed",type=int,default=42)
    main(p.parse_args())
