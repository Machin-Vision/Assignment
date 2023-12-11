import os
os.environ.pop("TFDS_DATA_DIR", None)

import tensorflow_datasets as tfds
dataset = tfds.data_source("oxford_iiit_pet", data_dir="../data", download=False)
