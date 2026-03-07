This directory contains code and data to reproduce the results for the aberration analysis from the section "*Aberration analysis*" of Methods and *Extended Data Figure 2*. <br>
In addition to the provided files, it also requires the *DAMN_ResNet.keras* model stored in the *ResNet* directory. <br>

-> **1_Generate_PSFs_with_aberrations.ipynb** <br>
A Jupyter notebook that generates realistic aberrated point spread functions using vectorial calculus with Zernike polynomials. <br>

-> **2_Generate_data_samples.ipynb** <br>
A Jupyter notebook that takes the generated point spread functions and generates simulated datasets for aberration analysis. <br>

-> **3_Evaluate.ipynb** <br>
A Jupyter notebook that takes the generated datasets and evaluates the DAMN ResNet model's robustness to aberrated PSFs. <br>

-> **func_file_Model.py** <br>
Architecture definitions for creating the DAMN ResNet model and subsequently loading the weights. <br>

-> **func_file_Aberrations.py** <br>
Supporting functions for the vectorial calculus with Zernike polynomials. <br>
