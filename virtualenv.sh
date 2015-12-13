module load virtualenv/12.1.1
cd $SCRATCH/reddit_classification
virtualenv env
source env/bin/activate

pip install --upgrade pip
pip wheel --wheel-dir=./wheelhouse pandas==0.16.2	
pip install --no-index --use-wheel --find-links=./wheelhouse pandas==0.16.2

pip freeze > requirements.txt
deactivate

module load virtualenv/12.1.1
pip install --upgrade -r requirements.txt


