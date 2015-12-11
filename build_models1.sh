#!/bin/bash

GDRIVE="../../Google Drive/gdrive"
DATA="${GDRIVE}/data"
W2V="${GDRIVE}/word2vec/"
D2V="${GDRIVE}/doc2vec/"
TRAIN="train2.txt"
TRAINPATH="${DATA}/${TRAIN}"
SENTENCEPATH="${GDRIVE}/data/train_w2v_ready.txt"
W2VMODEL="${W2V}/w2v_train_only.txt"
D2VMODEL="${D2V}/d2v_train_only_labels.txt"

python word2vec_split.py "$DATA" "$TRAIN" "$W2V"
python doc2vec_split.py "$TRAINPATH" "$SENTENCEPATH" "$D2V"
python check_models.py "$W2VMODEL" "$D2VMODEL"
