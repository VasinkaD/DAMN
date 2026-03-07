This directory contains code and data to reproduce the results for the binary stars from the section "*Astronomical and localization microscopy demonstration*" of the main text and *Figure 5*. <br>
In addition to the provided files, it also requires the *DAMN_ResNet.keras* model stored in the *ResNet* directory. <br>

-> **Binary_star_evaluate.ipynb** <br>
A Jupyter notebook that processes the astronomical data and visualizes its results. <br>

-> **func_file_Model.py** <br>
Architecture definitions for creating the DAMN ResNet model and subsequently loading the weights. <br>

-> **func_file_Binary_star.py** <br>
Supporting functions for processing the data. <br>

-> **Binary_star_LR.tiff** <br>
The low-resolution image from the ground-based telescope to be reconstructed. The fully unresolved binary stars are located in the center. <br>

-> **GT_table.txt** <br>
The ground-truth information on the binary stars from the space-based observatory provided by the Gaia Data Release 3. <br>
