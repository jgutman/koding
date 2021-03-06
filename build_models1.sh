#!/bin/bash

GDRIVE="../../Google Drive/gdrive"
DATA="${GDRIVE}/data"
W2V="${GDRIVE}/word2vec/"
D2V="${GDRIVE}/doc2vec/"
CLEAN="${GDRIVE}/data3.txt"
DIRTY="${GDRIVE}/data2.txt"
TRAIN="train2.txt"
TEST="test2.txt"
TRAINPATH="${DATA}/${TRAIN}"
TESTPATH="${DATA}/${TEST}"
SENTENCEPATH="${GDRIVE}/data/train_w2v_ready.txt"
W2VMODEL="${W2V}/w2v_train_only.txt"
D2VMODEL="${D2V}/d2v_train_only_labels.txt"

mkdir -p "$DATA"
mkdir -p "$W2V"
mkdir -p "$D2V"

python datacleanPartDeux.py "$DIRTY" "$CLEAN"
python split_updated.py "$CLEAN" "$TRAINPATH" "TESTPATH"
python word2vec_split.py "$DATA" "$TRAIN" "$W2V"
python doc2vec_split.py "$TRAINPATH" "$SENTENCEPATH" "$D2V"
python check_models.py "$W2VMODEL" "$D2VMODEL"
nohup python baseline_word2vec.py -w2v "$W2VMODEL" -train "$TRAINPATH" -test "$TESTPATH" -weighted -stopwords > w2v_fit.out
nohup python doc2vec_model.py -google "$GDRIVE" -doc2v "doc2vec/d2v_train_only_labels.txt" -data "data3.txt" -train "data/${TRAIN}" -test "data/${TEST}" -stopwords > d2v_fit.out

nohup python ../koding/doc2vec_split.py ./data/train2.txt ./doc2vec/d2v_train_only_tokenized.txt ./data/w2v_tokenized_train2.txt  > doc2v.train.only.tokenized.out

nohup python ../koding/doc2vec_split.py ./data/test2.txt ./doc2vec/d2v_test_only_tokenized.txt ./data/w2v_tokenized_test.txt  > doc2v.test.only.tokenized.out

nohup python ../koding/doc2vec_split.py ./data/data3.txt ./doc2vec/d2v_alldata_model_tokenized.txt ./data/w2v_tokenized_all_data.txt  > doc2v.alldata.tokenized.out


nohup python ../koding/baseline_word2vec.py -w2v './word2vec/w2v_train_only.txt' -train './data/train2.txt' -test './data/test2.txt' -weighted -stopwords -loadembeddings -storedvecpath './data' > w2v.loaded.fit.out

nohup python ../koding/baseline_word2vec.py -w2v './word2vec/w2v_train_only.txt' -train './data/train2.txt' -test './data/test2.txt' -weighted -stopwords -storedvecpath './data/w2vembeddings/' > w2v.weighted.fit.out

nohup python ../koding/baseline_word2vec.py -w2v './word2vec/w2v_train_only.txt' -train './data/train2.txt' -test './data/test2.txt' -stopwords -storedvecpath './data/w2vembeddings/unweighted' > w2v.unweighted.fit.out

nohup python ../koding/doc2vec_model.py -google './' -doc2v './doc2vec/d2v_train_only_labels.txt' -data './data/data3.txt' -train './data/train2.txt' -test './data/test2.txt' -stopwords -loadtest -testvecpath './doc2vec/test2.d2v.embeddings.pickle' > d2v.loaded.fit.out

python doc2vec_model.py -google '../../Google Drive/gdrive/' -doc2v './doc2vec/d2v_train_only_labels.txt' -data './data/data3.txt' -train './data/train2.txt' -test './data/test2.txt' -stopwords  -testvecpath './doc2vec/test2.d2v.embeddings.pickle'


python ../koding/evaluation.py -model baseline/predict_proba_log_LogisticRegression_baseline_ngram-1.csv  -saveoutput ./confusion_plots/baseline_LogisticRegression_ngram_1_confusion.png -header > ./confusion_plots/baseline_LogisticRegression_ngram_1.out