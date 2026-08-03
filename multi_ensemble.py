import argparse,os,numpy as np,torch,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from dataset_islr import GlossVocabISLR,load_metadata,get_splits
from models.fusion_model import build_fusion_model
from train_shubert_ft import SHFT,SUF
DIRMAP={"face":"face_feats","lh":"hand1_feats","rh":"hand2_feats","body":"body_feats"}
class DualDS(Dataset):
    def __init__(s,df,vc,pd,shb,mf=64):
        s.vc=vc;s.pd=pd;s.shb=shb;s.mf=mf;df=df.reset_index(drop=True)
        keep=[i for i,(_,r) in enumerate(df.iterrows()) if s._ok(r["video_id"])]
        s.df=df.iloc[keep].reset_index(drop=True);print(f"usable {len(s.df)}/{len(df)}")
    def _sp(s,k,v):return os.path.join(s.shb,DIRMAP[k],v+SUF[k]+".npy")
    def _pp(s,v):
        a=os.path.join(s.pd,v+".pt");return a if os.path.exists(a) else os.path.join(s.pd,v)
    def _ok(s,v):return os.path.exists(s._pp(v)) and all(os.path.exists(s._sp(k,v)) for k in SUF)
    def __len__(s):return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i];v=r["video_id"]
        pose=torch.load(s._pp(v),map_location="cpu",weights_only=True).float()[:s.mf]
        a={k:np.load(s._sp(k,v)) for k in SUF};T=min(min(len(a[k]) for k in SUF),s.mf)
        return {"pose":pose,"plen":pose.shape[0],**{k:torch.from_numpy(a[k][:T]).float() for k in SUF},
                "slen":T,"label":s.vc.encode(r["gloss"])}
def collate(B):
    mp=max(b["plen"] for b in B);ms=max(b["slen"] for b in B)
    pad=lambda x,m:torch.cat([x,torch.zeros(m-x.shape[0],x.shape[1])],0) if x.shape[0]<m else x
    o={"pose":torch.stack([pad(b["pose"],mp) for b in B]),"plen":torch.tensor([b["plen"] for b in B]),
       "slen":torch.tensor([b["slen"] for b in B]),"labels":torch.tensor([b["label"] for b in B])}
    for k in SUF:o[k]=torch.stack([pad(b[k],ms) for b in B])
    return o
def top(P,Y,ks=(1,5,10)):
    o=P.topk(max(ks),1).indices;return {k:100*(o[:,:k]==Y[:,None]).any(1).float().mean().item() for k in ks}
def main(a):
    dev=torch.device("cuda");asl=os.path.join(a.data_root,"data",os.environ.get("ISLR_DATASET","asl_citizen"))
    df=get_splits(load_metadata(os.path.join(asl,"metadata.csv")),seed=42)
    vc=GlossVocabISLR();vc.build(df[df.split=="train"]["gloss"].tolist())
    df=df[df["gloss"].isin(set(vc._gloss2id))].reset_index(drop=True);NC=len(vc)
    pd=os.path.join(asl,"pose_features");shb=os.path.join(asl,"shubert")
    va=DualDS(df[df.split=="val"],vc,pd,shb);te=DualDS(df[df.split=="test"],vc,pd,shb)
    dl=lambda ds:DataLoader(ds,batch_size=32,num_workers=0,collate_fn=collate)
    pose=build_fusion_model(num_classes=NC,pose_dim=261,rgb_dim=768,size="base",use_pose=True,use_rgb=False)
    pose.load_state_dict(torch.load(a.pose_ckpt,map_location="cpu")["model_state"]);pose.eval().to(dev)
    sh=SHFT(a.shubert_base,NC).to(dev);sh.load_state_dict(torch.load(a.shft_ckpt,map_location="cpu")["model_state"]);sh.eval()
    @torch.no_grad()
    def pr(ds):
        Pp=[];Ps=[];Y=[]
        for b in dl(ds):
            Pp.append(F.softmax(pose(pose_feats=b["pose"].to(dev),pose_lengths=b["plen"].to(dev))["logits"],-1).cpu())
            sb={k:b[k].to(dev) for k in SUF};sb["lengths"]=b["slen"]
            Ps.append(F.softmax(sh(sb)["logits"],-1).cpu());Y.append(b["labels"])
        return torch.cat(Pp),torch.cat(Ps),torch.cat(Y)
    print("val...");vp,vs,vy=pr(va);print("test...");tp,ts,ty=pr(te)
    best,ba=-1,0
    for i in range(21):
        al=i/20;m=top((1-al)*vp+al*vs,vy)[1];tm=top((1-al)*tp+al*ts,ty)[1];print(f"ALPHA_SWEEP alpha={al:.2f} val={m:.2f} test={tm:.2f}")
        if m>best:best,ba=m,al
    r=top((1-ba)*tp+ba*ts,ty)
    print(f"\nbest alpha(shubert)={ba:.2f} val_top1={best:.2f}")
    print(f"ENSEMBLE TEST: top1={r[1]:.2f} top5={r[5]:.2f} r10={r[10]:.2f}")
    print(f"  pose={top(tp,ty)[1]:.2f} | shubert_ft={top(ts,ty)[1]:.2f}")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_root",default="/project2/jessetho_1732/aditeya")
    p.add_argument("--pose_ckpt",default="/project2/jessetho_1732/aditeya/islr_pipeline/checkpoints_fusion/fusion_pose_base_seed42/best.pt")
    p.add_argument("--shft_ckpt",default="/project2/jessetho_1732/aditeya/islr_pipeline/ckpt_shubert_ft/best.pt")
    p.add_argument("--shubert_base",default="/project2/jessetho_1732/aditeya/data/shubert_weights/shubert.pt")
    main(p.parse_args())
