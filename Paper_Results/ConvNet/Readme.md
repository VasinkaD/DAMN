This directory contains the code and data files for the non-upsampling ConvNet model results. <br>
Each subdirectory uses the *DAMN_ConvNet.h5* model. Due to its size, it is available for download at [Zenodo repository](https://doi.org/10.5281/zenodo.14641651). <br>

-> **Simulated_data** <br>
Subdirectory with codes for generating simulated data, processing them using all methods, and visualizing the results. <br>
The results are characterized in the section "*Evaluation using simulated data*" of the main text and *Figure 2*. <br>

-> **Optical_experiment** <br>
Subdirectory with codes for processing the data samples measured with our optical setup, which allows full control over the ground-truth emitter distribution. <br>
The results are characterized in the section "*Optical microscopy experimental validation with full control over ground truth*" of the main text and *Figure 4*. <br>

-> **requirements_ConvNet.pip** <br>
A file containing the minimal list of packages and their required versions from pip freeze output for the Linux environment with Python 3.11.3 and CUDA 12.1.1.
