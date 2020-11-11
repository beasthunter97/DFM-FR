#!/bin/bash

sleep 1s

cd /home/mendel/coral/DFM_FR/

source /home/mendel/.bashrc
ssh -T git@gitlab.dfm-engineering.com
git pull > test.txt
