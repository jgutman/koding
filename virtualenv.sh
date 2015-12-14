module load virtualenv/12.1.1
cd $SCRATCH/reddit_classification
virtualenv env
source env/bin/activate

pip install --upgrade pip
pip wheel --wheel-dir=./wheelhouse pandas==0.16.2	
pip install --no-index --use-wheel --find-links=./wheelhouse pandas==0.16.2
pip wheel --wheel-dir=./wheelhouse gensim==0.12.3	
pip install --no-index --use-wheel --force-reinstall --find-links=./wheelhouse gensim==0.12.3
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

module load virtualenv/12.1.1
source env/bin/activate
pip install --upgrade -r requirements.txt

pip show argparse
pip show logging
pip show nltk
pip show scipy
pip show numpy
pip show gensim
pip show sklearn
