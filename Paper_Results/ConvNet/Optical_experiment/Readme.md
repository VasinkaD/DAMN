This directory contains codes and data for reproducing the results of the section "*Optical microscopy experimental validation with full control over ground truth*" of the main text and *Figure 4*. <br>
In addition to the provided files, it also requires the *DAMN_ConvNet.h5* model. Due to its size, it is available for download at XXX. <br>

-> **1_Generate_explike_data.ipynb** <br>
A Jupyter notebook that generates the simulated datasets for optical parameters that match the experimental setting. <br>
Namely, it generates sets for testing the dependence on the emitter concentration, with other optical parameters matched to the optical experiment. <br>

-> **2_Process_experiment.ipynb** <br>
Once the data has been generated, this Jupyter notebook processes the measured and simulated images using the DAMN ConvNet model and both Richardson-Lucy variants. Processing with Deep-STORM requires 
a standalone Python environment. <br>
As an additional note, the Richardson-Lucy algorithms can take substantial time for reconstructions (up to several hours, depending on the hardware). Therefore, the code includes an option to reproduce 
only a small portion of the data to speed up the process. <br>

-> **3_Visualize_results.ipynb** <br>
Once the data has been processed, the evaluated mean absolute error metric is loaded, and the graphs of *Figure 4* are visualized. <br>
The Deep-STORM results are loaded from the *DeepStorm_exp_MAEs.zip* file, as Deep-STORM requires a standalone environment. <br>

-> **func_file_Experiment.py** <br>
Supporting functions for generating data with optical parameters that match the experimental setting. <br>

-> **func_file_Process_exp.py** <br>
Supporting functions for processing the data with each method. For example, the definition of both Richardson-Lucy algorithms.

-> **DeepStorm_exp_MAEs.zip** <br>
The results obtained from all the device-dependent Deep-STORM models, as their evaluation requires a standalone environment.
