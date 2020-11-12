#!/bin/bash

source /home/mendel/.bashrc

cd /home/mendel/coral/DFM_FR/

# if ping -c 2 192.168.20.78 &> /dev/null
# then
#   c_cam1=true
# else
#   c_cam1=false
# fi

# if ping -c 2 192.168.20.80 &> /dev/null
# then
#   c_cam2=true
# else
#   c_cam2=false
# fi

# if $c_cam1 && $c_cam2
# then
#   status=1
# else
#   status=0
# fi

(python3 main_FR.py -s vid -v test1.m -d out > python_log.txt & (sleep 10s; git pull > update_log.txt))
