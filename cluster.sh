#!/bin/bash

#PBS -l mem=64GB
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=8
#PBS -M jg3862@nyu.edu
#PBS -m ae

source activate redditenv
cd /scratch/jg3862/gdrive

python ../koding/cluster_visualize.py