<b>Installation/Setup</b>:
If you use Anaconda, then I suggest creating a new environment with python 3.8.
In your command line: "<i>conda create -n NLradar python=3.8</i>".
Activate that environment with "<i>conda activate NLradar</i>".
Then go to directory in which you want to install NLradar. 
There, clone the git repository with "<i>git clone https://github.com/Bram94/NLradar.git</i>".
Then move into the NLradar folder where a setup.py script is located, and run "<i>pip install .</i>".
If the preceding steps went correctly, you should then be able to start the viewer from the NLradar/Python_files folder, by running "<i>python nlr.py</i>". 

Note: The Unet velocity dealiasing algorithm can run on either the CPU or GPU. The GPU option is used when you have an NVIDIA GPU on your PC, with CUDA and cuDNN installed. 
If not, the CPU is used, which is however much slower.

<b>Using the viewer</b>:
I suggest taking a look at the Help in order to get started. Navigation heavily uses the keyboard, and a list of all keyboard shortcuts can be found at the tab Help/Keyboard.

Note: This is a first release of NLradar. More expansive documentation will follow over time.

Created by Bram van 't Veen (2016-2024).

![image](https://github.com/Bram94/NLradar/assets/24604991/eabaeef3-5c1d-4561-9dd8-40d5007087d6)
![image](https://github.com/Bram94/NLradar/assets/24604991/8811126b-f541-4458-9597-34b033f20df2)



