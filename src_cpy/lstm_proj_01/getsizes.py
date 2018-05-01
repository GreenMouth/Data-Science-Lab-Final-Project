# Author John Sigmon
# Just grabbing sizes of my saved arrays

import numpy as np

data_name = 'data.npy'
labels_name = 'labels.npy'
metadata_name = 'metadata.npy'

data = np.load(data_name)
labels = np.load(labels_name)
metadata = np.load(metadata_name)

print("Size of transcript data is: {}".format(data.shape))
print("Size of labels is: {}".format(labels.shape))
print("Size of metadata data is: {}".format(metadata.shape))
