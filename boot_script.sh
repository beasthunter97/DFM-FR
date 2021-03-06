#!/bin/bash
source /home/mendel/.bashrc
cd /home/mendel/coral/DFM_FR/
mkdir -p temp/ log/
sudo mount -t ntfs-3g /dev/mmcblk1p1 ./temp/

(
    while true
    do
        count=`ping -c 1 8.8.8.8 | grep bytes | wc -l`
        if [ $count -gt 1 ]
        then
            break
        fi
    done
    commit=$(date)
    git pull > log/update_log.txt
    git add .
    git commit -m "$commit"
    git push
    count=`cat log/update_log.txt | grep "Already up to date" | wc -l`
    if [ $count -eq 0 ]
    then
        sudo reboot
    fi

    echo "false" > log/working
    sleep 60s
    working=$(cat log/working)
    if ! $working
    then
        sudo reboot
    fi
) &

sleep 5s
python3 main_FR.py

sudo umount ./temp/

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
