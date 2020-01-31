import tensorflow as tf
import tensorflow.keras.layers as layers



class constructCNN ( tf.keras.Model ):

    def __init__( self, num_actions, W   ):

        super( constructCNN, self).__init__()
        self.W              = W
        self.num_actions    = num_actions

        ### kernel size, pool size,  pool stride  ###
        kernel_size = 5 
        pool_size   = 2
        pool_stride = 2

        # maximum value of relu function
        ReLUMax     = 999

        # Layers
        self.L1     = layers.Conv2D     ( 8, (kernel_size,kernel_size), padding='same', input_shape=(W,W,1,) , use_bias=False )
        self.L1B    = layers.BatchNormalization()
        self.L1BR   = layers.ReLU(ReLUMax)

        self.L2     = layers.Conv2D     ( 8, (kernel_size,kernel_size), padding='same', use_bias=False )
        self.L2B    = layers.BatchNormalization()
        self.L2BR   = layers.ReLU(ReLUMax)
        self.L2BP   = layers.MaxPool2D  ( (pool_size,pool_size), (pool_stride, pool_stride) )

        self.L3     = layers.Conv2D     ( 16, (kernel_size,kernel_size), padding='same', use_bias=False )
        self.L3B    = layers.BatchNormalization()
        self.L3BR   = layers.ReLU(ReLUMax)

        self.L4     = layers.Conv2D     ( 16, (kernel_size,kernel_size), padding='same', use_bias=False )
        self.L4B    = layers.BatchNormalization()
        self.L4BR   = layers.ReLU(ReLUMax)
        self.L4BP   = layers.MaxPool2D  ( (pool_size,pool_size), (pool_stride, pool_stride) )
        self.L4BPF  = layers.Flatten    ()


        self.L5     = layers.Dense      ( 32, bias_initializer = tf.keras.initializers.TruncatedNormal (stddev=0.01) )
        self.L5B    = layers.BatchNormalization()
        self.L5BR   = layers.ReLU(ReLUMax)

        self.L6     = layers.Dense      ( num_actions, bias_initializer = tf.keras.initializers.TruncatedNormal (stddev=0.01) )
        self.L6S    = layers.Softmax    ()

    def call ( self, inputs, training ):

        # reshape
        x   = tf.reshape    ( inputs, [-1,self.W,self.W,1] )
        m,v = tf.nn.moments ( x, [0,1,2,3] )
        x   = self.normalize( x, m, v ) 

        # Layer 1
        x   = self.L1       ( x )
        x   = self.L1B      ( x, training= training )
        x   = self.L1BR     ( x )

        # Layer 2
        x   = self.L2       ( x )
        x   = self.L2B      ( x, training= training )
        x   = self.L2BR     ( x )
        x   = self.L2BP     ( x )

        # Layer 3
        x   = self.L3       ( x )
        x   = self.L3B      ( x, training= training )
        x   = self.L3BR     ( x )

        # Layer 4
        x   = self.L4       ( x )
        x   = self.L4B      ( x, training= training )
        x   = self.L4BR     ( x ) 
        x   = self.L4BP     ( x )
        x   = self.L4BPF    ( x )

        # Layer 5    
        x   = self.L5       ( x )
        x   = self.L5B      ( x, training= training )
        x   = self.L5BR     ( x )


        # Layer 6
        x   = self.L6       ( x )
        rho = self.L6S      ( x )

        return rho



    def normalize ( self, x, m, v ):

        if v == 0:
            return x

        return (x - m) / tf.sqrt( v ) 
