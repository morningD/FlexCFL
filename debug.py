import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from utils import test_trainer

print('This is a debug file')
