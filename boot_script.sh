#!/bin/bash

source /home/mendel/.bashrc
cd /home/mendel/coral/DFM_FR/
mkdir -p temp/ log/
sudo mount -t ntfs-3g /dev/mmcblk1p1 ./temp/

(
    echo "false" > log/working
    sleep 20s
    working=$(cat log/working)
    if ! $working
    then
        sudo reboot
    else
        git pull > log/update_log.txt
        git add .
        git commit -m "Upload log"
        git push >> log/update_log.txt
    fi
) &

sleep 3s
python3 main_FR.py -s vid -v test1.m4v -d out > log/python_log.txt

sudo umount /dev/mmcblk1p1

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
