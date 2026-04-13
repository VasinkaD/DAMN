This directory contains code and data to reproduce the results for the nuclear pore complexes dataset from the section "*Stellar astronomy and localization microscopy demonstration*" of the main text 
and *Figure 6-II*. <br>
In addition to the provided files, it also requires the *DAMN_ResNet.keras* model. Due to its size, it is available for download at [Zenodo repository](https://doi.org/10.5281/zenodo.14641651). <br>
Additionally, the nuclear pore complexes dataset, "*Gettingstarted2D_Nup96-AF647.zip*," is available at [this repository](https://www.embl.de/download/ries/Example_NPC3D2C/). <br>

-> **Process_NPC.ipynb** <br>
A Jupyter notebook that processes the nuclear pore complexes dataset and visualizes its results. <br>

-> **func_file_Model.py** <br>
Architecture definitions for creating the DAMN ResNet model and subsequently loading the weights. <br>

-> **func_file_NPC.py** <br>
Supporting functions for processing data. <br>

-> **DS_single_NPC.npy** <br>
The reconstruction of a single NPC using 5,000 low-resolution frames by the device-dependent Deep-STORM model, as its evaluation requires a standalone environment. Additionally, the Deep-STORM training and evaluation uses the codebase of its authors, which we do not intend to replicate in our repository.
