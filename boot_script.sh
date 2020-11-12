#!/bin/bash

sleep 3s

cd /home/mendel/coral/passenger_counting/

if ping -c 2 8.8.8.8 &> /dev/null
then
  c_int=true
else
  c_int=false
fi

if ping -c 2 192.168.20.78 &> /dev/null
then
  c_cam1=true
else
  c_cam1=false
fi

if ping -c 2 192.168.20.80 &> /dev/null
then
  c_cam2=true
else
  c_cam2=false
fi

if $c_int && $c_cam1 && $c_cam2 ;
then
  status=1
else
  status=0
fi

source /home/mendel/.bashrc

git pull

nohup /usr/bin/python3 /home/mendel/coral/passenger_counting/main_final_counter_data.py --status $status >/dev/null 2>&1 &


