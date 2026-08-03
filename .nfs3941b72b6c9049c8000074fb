import argparse, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset_fusion import build_fusion_datasets, collate_fusion
from dataset_islr import make_class_balanced_sampler
from models.fusion_model import build_fusion_model
from train_fusion import evaluate
from train_islr import topk_accuracy, cosine_warmup_schedule

def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device("cuda")
    sd=os.path.join(a.save_dir,f"cofusion_seed{a.seed}"); os.makedirs(sd,exist_ok=True)
    tr,va,te,vocab=build_fusion_datasets(a.data_root,use_pose=True,use_rgb=True,
        rgb_dir=a.rgb_dir,max_frames=a.max_frames,seed=a.seed)
    NC=len(vocab); vocab.save(os.path.join(sd,"vocab.json"))
    samp=make_class_balanced_sampler(tr.df,vocab)
    dl=lambda ds,s=None,bs=None:DataLoader(ds,batch_size=bs or a.batch_size,sampler=s,
        shuffle=(s is None and ds is tr),num_workers=a.num_workers,collate_fn=collate_fusion,
        pin_memory=True,multiprocessing_context="forkserver")
    trdl,vadl,tedl=dl(tr,samp),dl(va,bs=a.batch_size*2),dl(te,bs=a.batch_size*2)
    model=build_fusion_model(num_classes=NC,pose_dim=261,rgb_dim=768,size=a.size,
        use_pose=True,use_rgb=True,dropout=a.dropout).to(dev)
    opt=AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd)
    sched=cosine_warmup_schedule(opt,a.warmup_epochs,a.epochs); scaler=GradScaler("cuda")
    best=0.0; pat=0
    for ep in range(1,a.epochs+1):
        model.train(); t0=time.time()
        for b in trdl:
            pf,pl=b["pose_feats"].to(dev),b["pose_lengths"].to(dev)
            rf,rl=b["rgb_feats"].to(dev),b["rgb_lengths"].to(dev)
            y=b["labels"].to(dev); opt.zero_grad()
            with autocast("cuda"):
                o=model.forward_cotrain(pf,pl,rf,rl,labels=y)
                loss=(F.cross_entropy(o["logits"],y,label_smoothing=0.1)
                      +a.aux_w*F.cross_entropy(o["aux_pose"],y,label_smoothing=0.1)
                      +a.aux_w*F.cross_entropy(o["aux_rgb"],y,label_smoothing=0.1))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update()
        sched.step()
        m=evaluate(model,vadl,dev,"val")
        print(f"Ep {ep:3d}/{a.epochs} | val top-1={m['top1']:.2f} top-5={m['top5']:.2f} R@10={m['recall10']:.2f} | {time.time()-t0:.0f}s")
        if m["top1"]>best:
            best=m["top1"]; pat=0
            torch.save({"model_state":model.state_dict()},os.path.join(sd,"best.pt"))
            print(f"  ✓ best {best:.2f}")
        else:
            pat+=1
            if pat>=a.patience: print("early stop"); break
    model.load_state_dict(torch.load(os.path.join(sd,"best.pt"))["model_state"])
    t=evaluate(model,tedl,dev,"test")
    print(f"\nCOFUSION_RESULT: seed={a.seed} test_top1={t['top1']:.2f} test_top5={t['top5']:.2f} test_r10={t['recall10']:.2f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rgb_dir",required=True); p.add_argument("--data_root",default="/project2/jessetho_1732/aditeya")
    p.add_argument("--save_dir",default="checkpoints_cofusion"); p.add_argument("--size",default="base")
    p.add_argument("--epochs",type=int,default=120); p.add_argument("--warmup_epochs",type=int,default=5)
    p.add_argument("--batch_size",type=int,default=128); p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--wd",type=float,default=1e-4); p.add_argument("--dropout",type=float,default=0.2)
    p.add_argument("--max_frames",type=int,default=64); p.add_argument("--patience",type=int,default=20)
    p.add_argument("--num_workers",type=int,default=8); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--aux_w",type=float,default=0.3)
    main(p.parse_args())
