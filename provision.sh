export VHOME=/home/vagrant
wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $VHOME/miniconda.sh
chmod +x $VHOME/miniconda.sh
$VHOME/miniconda.sh -b -f -p $VHOME/miniconda
export PATH=$VHOME/miniconda/bin:$PATH
echo "export PATH=$VHOME/miniconda/bin:$PATH" >> $VHOME/.bashrc

conda create --yes -n py3opencv numpy scipy matplotlib pillow netcdf4 gdal opencv notebook
source activate py3opencv
echo "source activate py3opencv" >> $VHOME/.bashrc
pip install https://github.com/nansencenter/nansat/archive/v1.1.3.tar.gz
cd /vagrant
python setup.py install

echo "#!/bin/sh -e
exec 1>/tmp/rc.local.log 2>&1  # send stdout and stderr from rc.local to a log file
set -x
su vagrant -c \"/home/vagrant/miniconda/envs/py3opencv/bin/jupyter notebook --no-browser --notebook-dir=/vagrant --NotebookApp.token='' --ip=0.0.0.0\" &
" > /etc/rc.local
chown vagrant:vagrant $VHOME -R
su vagrant -c "/home/vagrant/miniconda/envs/py3opencv/bin/jupyter notebook --no-browser --notebook-dir=/vagrant --NotebookApp.token='' --ip=0.0.0.0" &
