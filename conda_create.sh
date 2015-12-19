bash $SCRATCH/reddit_classification/Miniconda-1.6.0-Linux-x86_64.sh

# [set installation directory to /archive/jg3862/anaconda2]
# [prepend path to .bashrc file]
# [restart terminal]

conda create --name venv python
conda update conda
source activate venv
conda list

conda install numpy
