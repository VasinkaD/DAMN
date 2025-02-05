# From Stars to Molecules: AI Guided Device-Agnostic Super-Resolution Imaging

This repository provides data and supplementary material for the paper **From Stars to Molecules: AI Guided Device-Agnostic Super-Resolution Imaging**, by Dominik Vašinka, Filip Juráň, Jaromír Běhal, and Miroslav Ježek. <br>
The paper is currently being prepared and will be available on arXiv. <br>
The repository is currently being developed; please await its completion.

In addition to this GitHub repository, the device-agnostic model and an experimental dataset are stored at the [Zenodo repository](https://zenodo.org/records/14641652) due to their size.

## The repository structure:
### Figure 2
The "Figure 2" folder contains Jupyter notebooks, and a supporting file with the Python function definitions, necessary to recreate the graph depicted in Fig. 2 of the publication. I.e., the dependence of mean absolute error on the emitter power, the width of the point spread function, the concentration of emitters, and the point spread function shape. <br><br>
Namely:
- "Fig_2_Data_generation.ipynb" generates and saves a simulated dataset for each panel of the Fig. 2
- "Fig_2_Method_evaluation.ipynb" uses these generated datasets to evaluate the performance of the deep learning model and the Richardson-Lucy algorithm
- "Fig_2_Results_vizualization.ipynb" visualizes the results of the previous notebook in the publication form
- "F2_func_file.py" contains the definitions of functions called by the three Jupyter notebooks
<br>
Moreover, these Jupyter notebooks require the device-agnostic learning model "DAMN_model.h5" stored at the Zenodo repository due to its size.

### Figure 4
The "Figure 4" folder contains Jupyter notebooks, and a supporting file with the Python function definitions, necessary to recreate the graph depicted in Fig. 4 of the publication. I.e., evaluation of the experimental data and comparison to simulated samples of the same optical parameters. <br><br>
Namely:
- "Fig_4_Data_generation.ipynb" generates and saves a simulated dataset using the optical parameter values corresponding to the experimental setup
- "Fig_4_Method_evaluation.ipynb" uses this generated dataset to evaluate the performance of the deep learning model and the Richardson-Lucy algorithm
- "Fig_4_Results_vizualization.ipynb" visualizes the results of the previous notebook in the publication form
- "F4_func_file.py" contains the definitions of functions called by the three Jupyter notebooks
<br>
Moreover, these Jupyter notebooks require the device-agnostic learning model "DAMN_model.h5" and the measured dataset from the experimental setup "Measured_data.npz" stored at the Zenodo repository due to their sizes.

### Figure 5
The "Figure 5" folder contains several files necessary to recreate the graph depicted in Fig. 5 of the publication. I.e., the demonstration of the device-agnostic model on the NGC 300 spiral galaxy image. <br><br>
Namely:
- "Fig_5_Data_processing.ipynb" contains the code for data pre-processing, evaluation, and results visualization corresponding to Fig. 5 of the publication
- "F5_func_file.py" contains the definitions of functions called by the Jupyter notebook
- "Upsampling_model.h5" is the adjusted device-agnostic model with implemented upsampling layers <br>

Moreover, the Jupyter notebook requires the original file "eso1037a.tif" of the spiral galaxy image. This 106 MB file can be downloaded as a "Fullsize Original" from the European Southern Observation [ESO](https://eso.org/public/images/eso1037a/) repository.

### Figure 6
The "Figure 6" folder contains several files necessary to recreate the graph depicted in Fig. 6 of the publication. I.e., the demonstration of the device-agnostic model on a high-concentration tubulin dataset from the single-molecule localization microscopy challenge. <br><br>
Namely:
- "Fig_6_Data_processing.ipynb" contains the code for data pre-processing, evaluation, and results visualization corresponding to Fig. 6 of the publication
- "F6_func_file.py" contains the definitions of functions called by the Jupyter notebook
- "Upsampling_model.h5" is the adjusted device-agnostic model with implemented upsampling layers
- "sequence.zip" contains the high-concentration tubulin data from [the single-molecule localization microscopy challenge](https://srm.epfl.ch/srm/dataset/challenge-2D-real/Real_High_Density/index.html)
- "SOSplugin_hd_image.png" is a reference image provided by the [SOSplugin](https://smal.ws/wp/software/sosplugin/)
<br>
This folder does not require any additional files.
