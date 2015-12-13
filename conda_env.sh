cp /etc/skel/.bashrc $HOME/
cp /etc/skel/.bash_profile $HOME/
source /home/jg3862/.bash_profile
source /home/jg3862/.bashrc

rm -rf $HOME/anaconda
cd $SCRATCH/reddit_classification

bash Miniconda-1.6.0-Linux-x86_64.sh
source $HOME/.bashrc

conda update conda

conda create --name redditenv gensim 
conda env list

source activate redditenv
conda install scikit-learn
conda list -e > ./redditenv_conda.txt
source deactivate

conda remove --name redditenv  --all
conda create -f --name redditenv --file ./redditenv_conda.txt  


