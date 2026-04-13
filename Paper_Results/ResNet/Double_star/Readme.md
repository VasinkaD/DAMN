This directory contains code and data to reproduce the results for the double star from the section "*Stellar astronomy and localization microscopy demonstration*" of the main text and *Figure 5*. <br>
In addition to the provided files, it also requires the *DAMN_ResNet.keras* model. Due to its size, it is available for download at [Zenodo repository](https://doi.org/10.5281/zenodo.14641651). <br>

-> **Double_star_evaluate.ipynb** <br>
A Jupyter notebook that processes the astronomical data and visualizes its results. <br>

-> **func_file_Model.py** <br>
Architecture definitions for creating the DAMN ResNet model and subsequently loading the weights. <br>

-> **func_file_Double_star.py** <br>
Supporting functions for processing the data. <br>

-> **Double_star_LR.tiff** <br>
The low-resolution image from the ground-based telescope to be reconstructed. The fully unresolved double stars are located in the center. <br>

-> **GT_table.csv** <br>
The ground-truth information on the double stars from the space-based observatory provided by the Gaia Data Release 3. <br>
