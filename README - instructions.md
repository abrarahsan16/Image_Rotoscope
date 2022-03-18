# ELE882 Final Project

Use this template to get started with the ELE882 final project.  It contains:

 * `make_cartoon.py`
    * An empty Python console application.
 * `cartoon_effect`
    * An empty Python module where you should place the bulk of your code.
 * `samples`
    * A few sample images for you to work with.  You may include other images if
      you wish.

## Requirements

The implementation **must** meet the following requirements:

 * There is a class called `CartoonEffect` in the `cartoon_effect` module.
    * All of the algorithm constants are set by the `CartoonEffect` initializer.
    * A user of the class may change a default by passing in a keyword argument
      into the class initializer.
    * The class has a method called `apply` that accepts an image and returns
      the stylized image.
 * The `make_cartoon.py` console application accepts the path to an existing
   image and the path to where the cartoon output will go.
 * Optionally, the `make_cartoon.py` script may also accept command line options
   that control the appearance of the stylized image.

As a concrete example, a user of the `cartoon_effect` library should be able to
do something similar to the code sample below.

```py
from skimage.io import imread
from cartoon_effect import CartoonEffect

img = imread(path_to_image)

# The effect will be initialized with all algorithm constants set to "good
# enough" defaults.
effect = CartoonEffect()
out = effect.apply(img)

# Alternatively, the effect may be initialized with different constants.
effect = CartoonEffect(some_constant=123)
out = effect.apply(img)
```
