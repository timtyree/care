#!/bin/bash
#Visualizes one action potential from the model following Luo and Rudy (1990)
# modified to support spiral defect chaos, as described in Qu (2000).
#Fortran implementation modified from the work of Prof.
# Wouter-Jan Rappel, my advisor.
# gfortran gener_table.f -o gener_table.x
# echo "Example Input: 1., 1., 0.1"
# ./gener_table.x
gfortran lr_0d.f -o lr_0d.x
./lr_0d.x
./xm.sh fort.1
