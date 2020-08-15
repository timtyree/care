#!/bin/bash
grep "Printing Outputs" Log/job.out.*.* | sort -n -k3 -r
tar -czvf Log.tar.gz Log
