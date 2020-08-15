Atrial Fibrillation Simulation
Computing the Birth-Death Rates of Spiral Tips with a Novel Tip Detection Method
Tim Tyree
UCSD - Rappel Lab
8.14.2020

The following is a how-to-use-osg-guide for present Tim by past Tim.
<!-- #log onto osg connect (which has nice tutorials) -->
$ ssh TimtheTyrant@login05.osgconnect.net
#enter password (hint:check phone, it's same as for cmail pass)


<!-- copy file using scp to the open science grid -->
$ scp -r pde-sim-transfer TimtheTyrant@login05.osgconnect.net:~
<!-- pw hint english cmail -->
<!-- copy file using scp from the open science grid -->
$ scp TimtheTyrant@login05.osgconnect.net:pde-sim-transfer/pde-sim.submit archive

<!-- copy output from osg  -->
$ scp -r TimtheTyrant@login05.osgconnect.net:pde-sim-transfer/Log data/osg_output

<!-- debug an already submitted job -->
$ condor_q
$ condor_q -better-analyze
$ nano broken_file.submit
$ condor_rm 3060807 #removes old jobs

<!-- test pde_sim_worker.py -->
$ chmod +x pde_sim_worker.py
$ ./pde_sim_worker.py 1 2 3 4 5 0.005 10


<!-- HTCondor Workflow for a given .submit file -->
Let us submit the job listed in the .submit file.
$ condor_submit pde-sim.submit

    $ condor_submit ScalingUp-PythonCals.submit
    Submitting job(s)..........
    10 job(s) submitted to cluster 329837.

Apply your `condor_q` and `connect watch` knowledge to see this job progress. After all
jobs finished, execute the `post_script.sh  script to sort the results.

    ./post_process.sh
    or
    ./post_script.sh

<!--queue command tricks-->
<!-- arguments = $(x_low) $(x_high) $(y_low) $(y_high)
# Queue command  list
queue x_low, x_high, y_low, y_high from (
-9 9 -9 9
-8 8 -8 8
-7 7 -7 7
-6 6 -6 6
-5 5 -5 5
-4 4 -4 4
-3 3 -3 3
-2 2 -2 2
-1 1 -1 1
)

# Queue from file
Queue from <filename>

# Queue outside .submit file without a Queue call
condor_submit cook.sub -queue in *.dat
or
dir /b *.dat | condor_submit cook.sub -que from -
 -->

<!--  Make a new virtual environment -->
#load python
$ module load python/3.7.0
#create a virtual environment
$ python3 -m venv my_env
#activate an environment and install packages to it
$ source my_env/bin/activate
(my_env) $ pip install numpy
#install all such environments as needed, then exit the virtual environment with
$ deactivate
#compress the directory
$ tar czf my_env.tar.gz my_env
