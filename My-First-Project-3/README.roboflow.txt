
My First Project - v3 2025-02-07 9:37pm
==============================

This dataset was exported via roboflow.com on May 26, 2025 at 3:58 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 362 images.
Objects are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -13 and +13 degrees
* Random shear of between -9° to +9° horizontally and -9° to +9° vertically
* Random Gaussian blur of between 0 and 0.6 pixels
* Salt and pepper noise was applied to 0.49 percent of pixels


