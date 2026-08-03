import os, glob, subprocess
from imageio_ffmpeg import get_ffmpeg_exe
FF=get_ffmpeg_exe()
VID=os.environ["VID"]; WORK=os.environ["WORK"]
BASE=os.path.join(os.environ["ISLR"],"fig_assets"); os.makedirs(BASE,exist_ok=True)
cand=[]
for f in glob.glob(os.path.join(WORK,"hand_crops","*_hand1.mp4")):
    b=os.path.basename(f)[:-10]
    h2=os.path.join(WORK,"hand_crops",b+"_hand2.mp4"); fc=os.path.join(WORK,"face_crops",b+"_face.mp4")
    raw=[p for e in (".mp4",".MOV",".mov",".webm") if os.path.exists(p:=os.path.join(VID,b+e))]
    if raw and os.path.exists(h2) and os.path.exists(fc):
        cand.append((min(os.path.getsize(f),os.path.getsize(h2),os.path.getsize(fc)), b, raw[0]))
cand.sort(reverse=True)
ids=cand[:4]
print("chosen ids:", [c[1] for c in ids])
def grab(src,dst,n):
    subprocess.run([FF,"-y","-loglevel","error","-i",src,"-vf",f"select=gte(n\\,{n})","-vframes","1",dst],check=False)
    return os.path.getsize(dst) if os.path.exists(dst) else 0
import mediapipe as mp, numpy as np
from PIL import Image
mpd=mp.solutions.drawing_utils; mph=mp.solutions.holistic
POSE_C=mph.POSE_CONNECTIONS; HAND_C=mp.solutions.hands.HAND_CONNECTIONS
hol=mph.Holistic(static_image_mode=True, model_complexity=1)
for _,vid_id,RAW in ids:
    d=os.path.join(BASE,vid_id); os.makedirs(d,exist_ok=True)
    for i,n in enumerate([4,14,24,34,44]):
        grab(RAW, f"{d}/frame{i}.png", n)
    for name,src in [("face",f"{WORK}/face_crops/{vid_id}_face.mp4"),
                     ("hand1",f"{WORK}/hand_crops/{vid_id}_hand1.mp4"),
                     ("hand2",f"{WORK}/hand_crops/{vid_id}_hand2.mp4")]:
        bs=0
        for n in (10,20,30,40):
            t=f"/tmp/_c{n}.png"; s=grab(src,t,n)
            if s>bs: bs=s; subprocess.run(["cp",t,f"{d}/{name}.png"])
    if os.path.exists(f"{d}/frame2.png"):
        img=np.array(Image.open(f"{d}/frame2.png").convert("RGB")); canvas=img.copy()
        res=hol.process(img)
        mpd.draw_landmarks(canvas,res.pose_landmarks,POSE_C)
        mpd.draw_landmarks(canvas,res.left_hand_landmarks,HAND_C)
        mpd.draw_landmarks(canvas,res.right_hand_landmarks,HAND_C)
        Image.fromarray(canvas).save(f"{d}/keypoints.png")
    print(vid_id, "->", sorted(os.listdir(d)))
hol.close(); print("ALL IN:", BASE)
