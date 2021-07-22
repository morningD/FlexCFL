import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from utils import test_trainer
#import importlib

print('This is a debug file')