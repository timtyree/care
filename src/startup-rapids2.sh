#after completely restarting my UB
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    rapidsai/rapidsai:21.06-cuda11.2-runtime-ubuntu20.04-py3.8
# docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
#     rapidsai/rapidsai:21.06-cuda11.0-runtime-ubuntu18.04-py3.7
#
# # then enter
# # bash /rapids/utils/start-jupyter.sh
