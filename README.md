# Semantic Segmentation for Automatic Extraction of Linear Geological Structures from UAVImagery - SAELGS

This Repository contains the code for the paper Semantic Segmentation for Automatic Extraction of Linear Geological Structures from UAV Imagery, including ground truth creation, segmentation and post processing. The `segmentation` folder contains the code for training a segmentation model, an adapted fork of https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-with-pytorch, and other code for training different segmentation models.

## Notebooks

There are several  `jupyter` notebooks containing several notebooks containing experiments with computer vision and data visualization used in the development of the submitted work. These still lack organization but are fully functional. If you need the to use as references for your work, please email me at `davibortolotti@gmail.com`.

## How to use

This code requires the creating of a virtual environment using Python v3.7.8. Then run `pip install -r requirements.txt` to install all dependencies. In order to use azimuth analysis, you will need to also install GDAL. I recommend installing anaconda for this.


## How to cite

If you use this code in the development of your work, please reference this github and the SAELGS paper (still unsibmitted at this point).
