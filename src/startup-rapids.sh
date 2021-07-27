DIR=~/Documents/GitHub/care/notebooks
cd $DIR
# docker pull rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.7
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.7 -v $DIR:/rapids/notebooks
jupyter notebook list
