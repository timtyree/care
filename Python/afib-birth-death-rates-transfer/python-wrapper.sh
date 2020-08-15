#!/bin/bash

module load python/3.7.0

# Unpack your envvironment (with your packages), and activate it
tar -xzf ttt_env.tar.gz
python3 -m venv ttt_env
source ttt_env/bin/activate

python3 ./pde_sim_worker.py  $1 $2 $3 $4 $5 $6 $7

deactivate
