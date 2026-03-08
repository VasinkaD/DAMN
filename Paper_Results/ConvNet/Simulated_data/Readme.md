This directory contains code and data for reproducing the results of the section "*Evaluation using simulated data*" of the main text and *Figure 2*. <br>
In addition to the provided files, it also requires the *DAMN_ConvNet.h5* model. Due to its size, it is available for download at [Zenodo repository](https://doi.org/10.5281/zenodo.14641651). <br>

-> **1_Generate_data.ipynb** <br>
A Jupyter notebook that generates the simulated datasets for various optical parameters. <br>
Namely, it generates the sets for testing the dependence on the signal-to-noise ratio, the point-spread function width, the emitter concentration, and the transition between Gaussian and Airy PSF profiles. <br>

-> **2_Process_data.ipynb** <br>
Once the data has been generated, this Jupyter notebook processes the images using the DAMN ConvNet model and both Richardson-Lucy variants. Processing with Deep-STORM requires a standalone Python environment. <br>
As an additional note, the Richardson-Lucy algorithms can take substantial time for reconstructions (up to days, depending on the hardware). Therefore, the code includes an option to reproduce only a small portion of the simulated data to speed up the process. <br>

-> **3_Visualize_results.ipynb** <br>
Once the data has been processed, the evaluated mean absolute error metric is loaded, and the graphs of *Figure 2* are visualized. <br>
The Deep-STORM results are loaded from the *DeepStorm_MAEs.zip* file, as Deep-STORM requires a standalone environment. <br>

-> **func_file_Simulate.py** <br>
Supporting functions for generating data with specified optical parameters. <br>
Used for generating data for the parameter-dependence testing. <br>

-> **func_file_Process.py** <br>
Supporting functions for processing the data with each method. For example, the definition of both Richardson-Lucy algorithms. <br>

-> **DeepStorm_MAEs.zip** <br>
The results obtained from all the device-dependent Deep-STORM models, as their evaluation requires a standalone environment. Additionally, the Deep-STORM training and evaluation uses the codebase of its authors, which we do not intend to replicate in our repository.
