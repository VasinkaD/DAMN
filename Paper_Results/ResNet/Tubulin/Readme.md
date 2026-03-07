This directory contains code and data to reproduce the results for the tubulin dataset from the section "*Astronomical and localization microscopy demonstration*" of the main text and *Figure 6-I*. <br>
In addition to the provided files, it also requires the *DAMN_ResNet.keras* model stored in the *ResNet* directory. <br>

-> **Tubulin_evaluate.ipynb** <br>
A Jupyter notebook that processes the tubulin dataset and visualizes its results. <br>

-> **func_file_Model.py** <br>
Architecture definitions for creating the DAMN ResNet model and subsequently loading the weights. <br>

-> **func_file_Process.py** <br>
Supporting functions for processing the data with each method. For example, the definition of both Richardson-Lucy algorithms.

-> **DeepStorm_MAEs.zip** <br>
The results obtained from all the device-dependent Deep-STORM models, as their evaluation requires a standalone environment.
