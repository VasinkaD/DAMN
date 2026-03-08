This directory contains code and data to reproduce the results for the tubulin dataset from the section "*Astronomical and localization microscopy demonstration*" of the main text and *Figure 6-I*. <br>
In addition to the provided files, it also requires the *DAMN_ResNet.keras* model. Due to its size, it is available for download at XXX. <br>

-> **Tubulin_evaluate.ipynb** <br>
A Jupyter notebook that processes the tubulin dataset and visualizes its results. <br>

-> **func_file_Model.py** <br>
Architecture definitions for creating the DAMN ResNet model and subsequently loading the weights. <br>

-> **func_file_Tubulin.py** <br>
Supporting functions for processing data with each method and for evaluating quantitative results. <br>

-> **sequence.zip** <br>
Compressed stack of 500 low-resolution TIFF images that form the tubulin microscopy dataset. Also available from the website of [the single-molecule localization microscopy challenge](https://srm.epfl.ch/srm/dataset/challenge-2D-real/Real_High_Density/index.html) <br>

-> **DS_reconstruction.mat** <br>
The tubulin reconstruction provided directly by the authors of Deep-STORM, available in their [repository](https://github.com/EliasNehme/Deep-STORM). <br>

-> **SOS_detections.txt** <br>
The localization table provided by the [SOS Plugin](https://smal.ws/wp/software/sosplugin/), when applied to the tubuling dataset. <br>
