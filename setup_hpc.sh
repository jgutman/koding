ssh hpctunnel
ssh mercer
cd $SCRATCH

git clone --bare https://github.com/richardNam/koding.git
cd koding.git/
git push --mirror https://github.com/jgutman/koding.git
cd ..
rm -rf koding.git/

git clone https://github.com/jgutman/koding
cd koding
git pull
