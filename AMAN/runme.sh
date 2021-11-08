#!/bin/bash
sudo chmod 777 expt.py    #It will give the user permission(ie  to read write and execute) 
sudo apt-get install python  #It will install python in case you dont have it else it will get skipped
pip install opencv-python    #It will install the opencv package
pip install opencv-contrib-python-nonfree  #Some more packes which we will use in code
pip install numpy    #Will download numpy in case ypou dont have one
python3 expt.py       #Finally it will run the python code
