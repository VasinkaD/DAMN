This directory contains code and data for training a DAMN model using incremental learning. <br>
The chosen demonstration is the upsampling ResNet model, but the architecture can be easily replaced with other designs.

-> **DAMN_training.ipynb** <br>
A Jupyter notebook that covers everything for training a DAMN model. Starting from setting optical parameter ranges for simulated data, building the model architecture, to running the training loop. <br>
Training progress is saved along with logs and model checkpoints throughout training. <br>

-> **func_file_Data.py** <br>
Supporting functions for generating simulated data suitable for incremental learning using multiple processors in parallel with the *concurrent.futures.ProcessPoolExecutor*. <br>

-> **func_file_Model.py** <br>
Architecture definitions for building the DAMN ResNet model. <br>
