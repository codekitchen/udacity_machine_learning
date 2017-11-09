#!/bin/bash

sudo apt-get update
sudo apt-get install -y swig
sudo pip3 install 'gym[all]' pygame


git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
sudo pip3 install -e .
cd

git clone https://github.com/lusob/gym-ple.git
cd gym-ple/
sudo pip3 install -e .
cd

git clone git@github.com:codekitchen/udacity_machine_learning.git
echo "don't forget to use screen!"