cp /etc/skel/.bashrc $HOME/
cp /etc/skel/.bash_profile $HOME/
source /home/jg3862/.bash_profile
source /home/jg3862/.bashrc

# bash /scratch/jg3862/reddit_classification/Miniconda-1.6.0-Linux-x86_64.sh
bash /archive/jg3862/Anaconda2-2.4.1-Linux-x86_64.sh
# [set installation directory to /home/jg3862/anaconda2]
# [prepend path to .bashrc file]
# [restart terminal]

source /home/jg3862/.bashrc
conda update conda
conda create --name redditenv python

source activate redditenv
conda list
conda install numpy
conda install scipy
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install gensim
conda install nltk
conda list
conda list -e > /scratch/jg3862/reddit_classification/redditenv_conda.txt

python /scratch/jg3862/koding/test_import.py

source deactivate

# conda remove --name redditenv  --all
# conda create --name redditenv --file /scratch/jg3862/reddit_classification/redditenv_conda.txt