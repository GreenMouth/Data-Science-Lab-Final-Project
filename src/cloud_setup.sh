#! /usr/bin/env bash

# Author John Sigmon
# Last updated 4/24/18

GITDIR='Data-Science-Lab-Final-Project'
DESIREDDIR='dsl_final'

sudo apt-install git 2> /dev/null
sudo apt-install screen 2> dev/null
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
git clone https://github.com/jsigee87/$GITDIR.git
mv $GITDIR $DESIREDDIR
python3 -m virtualenv $DESIREDDIR
cd $DESIREDDIR


