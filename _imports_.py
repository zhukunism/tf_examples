from __future__ import print_function

import os
import numpy as np
import pandas as pd
import time

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.datasets import cifar10
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import Input, merge
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers.merge import add

import tensorflow as tf

from utils import *
