import numpy as np
import tensorflow as tf
import random as rn
import os
from keras import backend as K

random_seed = 42

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(random_seed)
rn.seed(random_seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(random_seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
