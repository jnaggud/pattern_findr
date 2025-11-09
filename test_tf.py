import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("TensorFlow successfully imported.")
