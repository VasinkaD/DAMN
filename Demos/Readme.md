This directory contains Jupyter notebooks and .py function files with basic demos on running the DAMN model.

1_Introduction.ipynb <br>
Covers proper model loading and verifies functionality by reconstructing a known sample. The result of the DAMN model application is compared against the result obtained by us, saved in the Pre-generated_test_sample.npz. <br>
Also includes basic data generation to apply the DAMN model to a stack of images with user-specified imaging parameters. <br>

2_Apply_to_loaded_data.ipynb <br>
A simple notebook to load the DAMN model and apply it to a user-specified dataset of images. <br>
Covers the proper data structure (shapes) and normalization for proper processing by the DAMN model. <br>
By default, it is demonstrated using a small subset (10 frames from the Small_demo_stack.zip) of the tubulin dataset. <br>

func_file_Data.py <br>
Supporting functions for tubulin data loading and simplified data generation. <br>

func_file_Model.py <br>
Architecture definitions to create the model and subsequently load the weights. <br>

Pre-generated_test_sample.npz <br>
A single simulated low-resolution image with its high-resolution ground truth and the expected result of the DAMN model. Serves to verify the proper functioning of the model on your device. <br>

Small_demo_stack.zip <br>
A small subset of the tubulin dataset to demonstrate the proper data structure and normalization.
