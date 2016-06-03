# Texture-Segmentation-Using-Texture-Filters-and-OpenCV
This is an adaptation of the texture segmentation, using texture filters method and OpenCV. The method is presented in Matlab code, in this excellent tutorial http://uk.mathworks.com/help/images/examples/texture-segmentation-using-texture-filters.html

### Image segmentation.
Image segmentation is a the process of partitioning an image into multiple segments. [Here](https://en.wikipedia.org/wiki/Image_segmentation) is the wikipedia article on Image segmentation.

### Texture segmentation
The featured method partitions an image based on texture. It is an adaptation of [this](http://uk.mathworks.com/help/images/examples/texture-segmentation-using-texture-filters.html) Mathworks example into OpenCV.

### Implemented functions
Some of the functions implemented as steps of the featured texture segmentation method are:
* entropyfilt: Returns a matrix, of the same size as the input image, where each pixel contains the entropy value of its 9x9 neighborhood.
* calculateEntropy: Calculates the entropy of an image, as described [here](http://uk.mathworks.com/help/images/ref/entropy.html)

### Borrowed functions
* bwareaopen: I found the code [here](http://opencv-code.com/quick-tips/code-replacement-for-matlabs-bwareaopen/)(I think) but I cannot access it any more. If someone knows about the contributor please let me know.
* imfill: The code comes from [here](http://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/).

