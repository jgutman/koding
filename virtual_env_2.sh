module purge
module load python/intel/2.7.6
module load virtualenv/12.1.1

# added by Miniconda 1.6.0 installer
export PATH="/archive/jg3862/anaconda/bin:$PATH"

cd $SCRATCH/koding
# rm -r wheelhouse
# rm -r venv
mkdir -p wheelhouse
mkdir -p venv

virtualenv venv
source venv/bin/activate 

pip install --upgrade pip
pip install wheel

# pip wheel --wheel-dir=./wheelhouse numpy
# pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse numpy

# pip wheel --wheel-dir=./wheelhouse scipy
# pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse scipy

pip wheel --wheel-dir=./wheelhouse pandas
pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse pandas

pip wheel --wheel-dir=./wheelhouse sklearn
pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse sklearn

pip wheel --wheel-dir=./wheelhouse matplotlib
pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse matplotlib

pip wheel --wheel-dir=./wheelhouse gensim
pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse gensim

pip freeze > requirements.txt
deactivate
module purge

source venv/bin/activate 
pip install --upgrade --no-index --use-wheel --find-links=./wheelhouse -r requirements.txt
python test_import.py
deactivate

