import numpy as np
import tensorflow as tf
import random 
import math 

class exRep:

    def __init__( self, M, width, height ) :
        self.M          = M
        self.W          = width
        self.H          = height

        self.curS       = list()    # listof Matrix
        self.curA       = list()    # listof lenth 3 onehot vector
        self.curR       = list()    # listof Scalar
        self.nxtS       = list()    # listof Matrix

        # No Terminal State


    def remember ( self, curS, curA, curR, nxtS ):

        # remember current experience
        self.curS.append   ( curS )
        self.curA.append   ( curA )
        self.curR.append   ( round( curR, 4)  )
        self.nxtS.append   ( nxtS )

        # delete oldest experience
        if( len( self.curS ) > self.M ):
            del self.curS[0]
            del self.curA[0]
            del self.curR[0]
            del self.nxtS[0]


    def get_Batch   ( self, CNN_T, Beta, numActions, Gamma ):

        curSs   = np.zeros( (Beta, self.W, self.H ) )
        curAs   = np.zeros( (Beta, numActions ) )
        nxtSs   = np.zeros( (Beta, self.W, self.H ) )
        Targets = np.zeros( (Beta, numActions ) )

        # get batchsize Beta random index from Memory Size List
        rIdxs   = random.sample( range( len( self.curS ) ), Beta )

        for k in range ( Beta ):

            curSs[k]    = self.curS[ rIdxs[k] ]
            curAs[k]    = self.curA[ rIdxs[k] ]
            nxtSs[k]    = self.nxtS[ rIdxs[k] ]

        rho = CNN_T ( tf.convert_to_tensor( nxtSs, dtype=tf.float32 ), False )
        for k in range (Beta):
            Targets[k][ np.argmax(curAs[k])]  = self.curR[ rIdxs[k] ] + Gamma * rho[k][np.argmax(rho[k])]

        return curSs, curAs, Targets
    
    
