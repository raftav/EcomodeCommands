# Avoid printing tensorflow log messages
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import time
import sys 
import freeze

#################################
# Useful constant and paths
#################################
ExpNum = 1
restore_epoch=31

result = freeze.freeze('model_exp'+str(ExpNum)+'_epoch'+str(restore_epoch),
						'/home/local/IIT/rtavarone/Ecomode/EcomodeItalianDeltasOnlyIREC/BiLSTM_export/',
						"model/SMO")