#!/bin/bash

#PBS -l mem=64GB
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=8
#PBS -M jg3862@nyu.edu
#PBS -m ae

source activate redditenv
cd /scratch/jg3862/gdrive

python ../koding/test_import.py

python ../koding/grid_search_d2v.py -context 15 -dims 100 150 200 250 300 400 500 600 -epochs 5 -cores 8 > d2vtune/logs/con15_dimall_epoch5.out

source deactivate