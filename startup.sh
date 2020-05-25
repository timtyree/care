#!/bin/bash

if [ jupyterlab -v ]; then
#echo "jupyterlab found"
else
echo "jupyterlab not found. installing jupyterlab..."
pip install jupyterlab
fi

if [ jupyterlab-nvdashboard -v ]; then
#echo "jupyterlab-nvdashboard found"
else
echo "jupyterlab-nvdashboard not found, installing jupyterlab_nvdashboard"
pip install jupyterlab-nvdashboard
jupyter labextension install jupyterlab-nvdashboard
jupyter lab build jupyterlab_nvdashboard
fi

export JUPYTERLAB_DIR = jupyter/rapidsainotebook
# export JUPYTERLAB_DIR = jupyter/rapidsai-notebook
# export JUPYTERLAB_DIR = jupyter/datasciencenotebook
#TODO: launch a particular workview of interest.  Hint:
#jupyter lab --help
#jupyter notebook --app-dir=~/jupyterlab_nvdashboard
#sudo nvidia-docker run --rm --name tf-notebook -p 8888:8888 -p 6006:6006 gcr.io/tensorflow/tensorflow:latest-gpu jupyter notebook --allow-root
python3 -c "from notebook.auth import passwd; print(passwd('your_password'))"
jupyter lab --app-dir ~/jupyter/jupyterlab_bokeh_server
jupyter lab --app-dir ~/jupyter/jupyterlab_bokeh_server --NotebookApp.password='sha1:2c8f1205d42b:82f4170455a19ad97ef3209be36603845a5c154e'

#TODO: google how to print the dependencies of at the end of a .ipynb
#TODO: $ grep ^that. record in a .sh file

jupyter lab
