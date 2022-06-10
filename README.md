# Graph-Based-Segmentation

Python implementation of [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf) by Felzenszwalb et. al.

## Usage
First use `requirements.txt` to install required packages.
Then you may use the tool as

```
main.py [-h] [--k K] [--sigma SIGMA] [--method {grid,nn}] [--num_neighbors NUM_NEIGHBORS]
```

### Mandatory Arguments:

`input` Image to be segmented  
`output` Segmented image output

### Optional Arguments:
`-h, --help` Show help message  
`--k K` Algorithm parameter indicating size of segments generated. Higher the k, larger the segments generated. Default = 1000  
`--sigma SIGMA` Sigma used by Gaussian blurring before running the segmentation algorithm. Default = 0 (Sigma calculated based on kernel size automatically)  
`--method {grid,nn}` Method to be used for generating graph, either grid graph or nearest neighbors in RGBXY space. Default = nn  
`--num_neighbors NUM_NEIGHBORS` Number of neighbors to be used to build grpah if NN method is used. Default = 8