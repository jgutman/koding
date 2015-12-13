#!/bin/bash

#PBS -l mem=8GB
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=1
#PBS -M jg3862@nyu.edu
#PBS -m ae

module load virtualenv/12.1.1
cd $SCRATCH/reddit_classification

rm -r ./wheelhouse/*
rm -r ./env/*

virtualenv env
source env/bin/activate

pip install --upgrade pip
pip install wheel

pip wheel --wheel-dir=./wheelhouse pandas==0.16.2	
pip install --no-index --force-reinstall --use-wheel --find-links=./wheelhouse pandas==0.16.2
pip wheel --wheel-dir=./wheelhouse gensim==0.12.3	
pip install --no-index --force-reinstall --use-wheel --find-links=./wheelhouse gensim==0.12.3
# pip wheel --wheel-dir=./wheelhouse numpy==1.10.1	
# pip install --no-index --use-wheel --find-links=./wheelhouse numpy==1.10.1
# pip wheel --wheel-dir=./wheelhouse scipy==0.16.0	
# pip install --no-index --use-wheel --find-links=./wheelhouse scipy==0.16.0
pip wheel --wheel-dir=./wheelhouse argparse	
pip install --no-index --use-wheel --find-links=./wheelhouse argparse
pip wheel --wheel-dir=./wheelhouse logging	
pip install --no-index --use-wheel --find-links=./wheelhouse logging
pip wheel --wheel-dir=./wheelhouse nltk	
pip install --no-index --use-wheel --find-links=./wheelhouse nltk
pip wheel --wheel-dir=./wheelhouse sklearn	
pip install --no-index --use-wheel --find-links=./wheelhouse sklearn

pip freeze > requirements.txt
deactivate