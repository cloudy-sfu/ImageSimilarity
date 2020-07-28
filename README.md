# ImageSimilarity

![](https://img.shields.io/badge/dependencies-tensorflow%201.15-orange)
![](https://img.shields.io/badge/dependencies-windows%2010-blue)

The program down-samples trademarks to 100\*100 pixels, process the images with enhancing, median blur and laplace filter to extract the edge of each colorblock of the trademarks.

Then, uses the siamese neural network to calculate the distance of two images in parameter space. The distance shorter than a threshold (determined by contrastive loss function with margin=1) means the two images are similar.

## Usage

Run `main.py` in master branch, and follow the guidance of GUI.
