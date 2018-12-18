#!/bin/sh
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
sudo pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
sudo apt-get install python-tk
chmod +x spectrogram/spectrogram.py
