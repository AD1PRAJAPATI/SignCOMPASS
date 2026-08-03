#!/bin/bash
# AUTSL bootstrap on CARC — run on the login node after you have the ChaLearn zips + keys.
# Puts data on scratch (quota-safe), then symlinks into the ISLR data root.
set -euo pipefail
SCRATCH=${SCRATCH:-/scratch1/aditeyak/autsl}
PROJ=${PROJ:-/project2/jessetho_1732/aditeya/data/autsl}
ISLR=${ISLR:-/project2/jessetho_1732/aditeya/islr_pipeline}

mkdir -p "$SCRATCH"/{train,val,test,raw}
mkdir -p "$(dirname "$PROJ")"

echo "1) Download ChaLearn AUTSL (encrypted) into $SCRATCH/raw/"
echo "   https://chalearnlap.cvc.uab.es/dataset/40/description/"
echo "   You need Codalab/ChaLearn registration for decryption keys."
echo "   Also grab SignList_ClassId_TR_EN.csv from the challenge page."
echo
echo "2) After zips + keys are in $SCRATCH/raw/, decrypt/extract, e.g.:"
echo "   cd $SCRATCH/raw && unzip train.zip && ...   # follow ChaLearn key instructions"
echo "   Arrange RGB only as:"
echo "     $SCRATCH/train/*_color.mp4"
echo "     $SCRATCH/val/*_color.mp4"
echo "     $SCRATCH/test/*_color.mp4"
echo "   plus train_labels.csv val_labels.csv test_labels.csv SignList_ClassId_TR_EN.csv"
echo
echo "3) Symlink into project data root (so ISLR_DATASET=autsl works):"
echo "   ln -sfn $SCRATCH $PROJ"
echo
echo "4) Build metadata:"
echo "   cd $ISLR && python autsl_make_metadata.py --root $PROJ"
echo
echo "5) Point videos at a flat videos_mp4 dir (optional helper after metadata exists)."
echo "DONE scaffold. Scratch dir: $SCRATCH"
ls -la "$SCRATCH"
