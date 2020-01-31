from __future__ import absolute_import, division, print_function
from operator import itemgetter 
import tensorflow as tf
import numpy as np


import experimentsRL as exRL
import experimentsSL as exSL

##############################################################################
### Tensorflow version 1.10  ###
print ( 'current tensorflow version :',tf.__version__)
##############################################################################
gpu_config = tf.ConfigProto()  
gpu_config.gpu_options.allow_growth = True 
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.enable_eager_execution( config=gpu_config  )


#################################################################################
TopN            = 1         # choose the best performing parameters in validation set
TRCost          = 0         # Transaction Cost
Tau_w           = 0         # Threshold used in laggedposchange. The default value is 0
Tau_s           = 0         # Threshold used in laggedposchange. The default value is 0
#################################################################################

############################ REINFORCEMENT LEARNING ############################## 
### HYPER-PARAMETERS  ### 
# The memory buffer size        : M     :   1000
# Iteration interval            : B     :   10
# Target network update interval: C     :   1000
# The batchzie                  : Beta  :   32
# Learningrate                          :   0.00001
# Action penalty                : P     :   0.1
paralistRL      = np.array( [1000, 10, 1000, 32,  0.00001, 0.1]  )
maxiterRL       = 3000001       # The number of maxmimum iteration  3000,001
ep_decay        = 0.999999      # The epsilon decay rate


start_experimentRL = exRL.startExperiment( 0 )
start_experimentRL.RL_MLP    ( TopN, maxiterRL, ep_decay, paralistRL, 'MR_p',  'MLP', 'VP',  TRCost, Tau_w, Tau_s  )
start_experimentRL.RL_MLP    ( TopN, maxiterRL, ep_decay, paralistRL, 'MR_pv', 'MLP', 'VPV', TRCost, Tau_w, Tau_s  )
start_experimentRL.RL_CNN    ( TopN, maxiterRL, ep_decay, paralistRL, 'CR_p',  'CNN', 'CP',  TRCost, Tau_w, Tau_s  )
start_experimentRL.RL_CNN    ( TopN, maxiterRL, ep_decay, paralistRL, 'CR_pv', 'CNN', 'CPV', TRCost, Tau_w, Tau_s  )



############################## SUPERVISED LEARNING ################################ 

### HYPER-PARAMETERS FOR SL ### 
# The batchzie                  : Beta  : 64
# Learningrate                          : 0.001
paralistSL  = np.array( [64, 0.001]  )
maxiterSL       = 200001           # The number of maxmimum iteration  

start_experimentSL = exSL.startExperiment( 0 )
start_experimentSL.SL_MLP    ( TopN, maxiterSL, paralistSL, 'MS_p',   'MLP', 'VP',  TRCost, Tau_w, Tau_s  )
start_experimentSL.SL_MLP    ( TopN, maxiterSL, paralistSL, 'MS_pv',  'MLP', 'VPV',  TRCost, Tau_w, Tau_s  )
start_experimentSL.SL_CNN    ( TopN, maxiterSL, paralistSL, 'CS_p',   'CNN', 'CP',  TRCost, Tau_w, Tau_s  )
start_experimentSL.SL_CNN    ( TopN, maxiterSL, paralistSL, 'CS_pv',  'CNN', 'CPV',  TRCost, Tau_w, Tau_s  )

















