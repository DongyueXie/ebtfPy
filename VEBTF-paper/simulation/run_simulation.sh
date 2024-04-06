 #!/bin/bash
# The script above runs the simulation with the following parameters: 

# reps : number of repetitions 
# sigma : noise level 
# n : number of samples 
# signal_name : name of the signal 
# file_name : name of the file 
# snr : signal-to-noise ratio 

# The script runs the simulation 20 times with a noise level of 1.0, 1024 samples, a signal named blocks, a file named blocks, and a signal-to-noise ratio of 3.0. 
# The script uses the  main.py  file in the  simulation  directory. 
# The  main.py  file is the main script that runs the simulation. 

reps=20
sigma=1.0
n=1024
signal_name="gauss"
file_name="gauss"
snr=3.0

python -u simulation/main.py --repetitions $reps --sigma $sigma --n $n --signal_name $signal_name --file_name $file_name --snr $snr
 
