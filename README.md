# SsdMultibox

This library provides utility classes for more easily performing [SSD Multibox Object Detection](https://arxiv.org/pdf/1512.02325.pdf) in [PyTorch](https://pytorch.org/)

# Data

Logic to read the [Pascal 2007](http://host.robots.ox.ac.uk/pascal/VOC/) for PyTorch `Dataset` and `DataLoader` is available.

The code expects to see this data in the `DATA_DIR` file directory:

- [Pascal VOC mirror](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Pascal JSON](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip)

Or get the data as a [Kaggle Dataset](https://www.kaggle.com/mikebaik/pascal-fastai-version)

# Run Project

The project uses Python 3.6

Install 3rd party libraries using:

`pip install -r requirements.txt`

# Run Tests

`py.test`

# License

MIT
