import tensorflow as tf
import tensorflow.keras.layers as layers



class constructDense  ( tf.keras.Model ):

    def __init__( self,num_actions, W  ):

        super( constructDense, self).__init__()
        self.W              = W
        self.num_actions    = num_actions

        # maximum value of ReLU
        ReLUMax     = 999

        # Layers
        self.L1     = layers.Dense      ( 64, bias_initializer = tf.keras.initializers.TruncatedNormal (stddev=0.01) )
        self.L1B    = layers.BatchNormalization()
        self.L1R    = layers.ReLU(ReLUMax )

        self.L2     = layers.Dense      ( 32, bias_initializer = tf.keras.initializers.TruncatedNormal (stddev=0.01) )
        self.L2B    = layers.BatchNormalization()
        self.L2R    = layers.ReLU(ReLUMax )

        self.L3     = layers.Dense      ( 16, bias_initializer = tf.keras.initializers.TruncatedNormal (stddev=0.01) )
        self.L3B    = layers.BatchNormalization()
        self.L3R    = layers.ReLU(ReLUMax )
        
        self.L4     = layers.Dense      ( num_actions, bias_initializer = tf.keras.initializers.TruncatedNormal (stddev=0.01) )

    def call ( self, inputs, training ):

        # reshape
        x   = tf.reshape    ( inputs, [-1,self.W] )
        m,v = tf.nn.moments ( x, [0,1] )
        x   = self.normalize( x, m, v ) 

        # Layer 1 
        x   = self.L1       ( x )
        x   = self.L1B      ( x, training= training )
        x   = self.L1R      ( x )

        # Layer 2
        x   = self.L2       ( x )
        x   = self.L2B      ( x, training= training )
        x   = self.L2R      ( x )
        
        # Layer 3
        x   = self.L3       ( x )
        x   = self.L3B      ( x, training= training )
        x   = self.L3R      ( x )

        # Layer 4
        rho = self.L4       ( x ) 

        return rho 



    def normalize ( self, x, m, v ):

        if v == 0:
            return x

        return (x - m) / tf.sqrt( v ) 
