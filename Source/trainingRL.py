import tensorflow as tf
import numpy as np
import random
import datetime 
import exReplay as exR
import shutil 
import os
from operator import itemgetter


class trainModel:

    def __init__   ( self,  epsilon_init, epsilon_min, epsilon_decay, num_actions, Gamma,  maxiter  ):
        self.num_actions    = num_actions
        self.maxiter        = maxiter
        
        self.epsilon        = epsilon_init 
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.Gamma          = Gamma

    def set_Data    ( self, TrX, TrY, ValX, ValY ):
        self.TrX    = TrX
        self.TrY    = TrY
        self.ValX    = ValX
        self.ValY    = ValY

        print 'Training X   :  Comp#, Days# ', len( self.TrX ), len( self.TrX[0] ), np.shape( self.TrX )
        print 'Training Y   :  Comp#, Days# ', len( self.TrY ), len( self.TrY[0] ), np.shape( self.TrY )
        print 'Validation X :  Comp#, Days# ', len( self.ValX ), len( self.ValX[0] ), np.shape( self.ValY )
        print 'Validation Y :  Comp#, Days# ', len( self.ValY ), len( self.ValY[0] ), np.shape( self.ValY )

    def get_LossR        ( self, Beta, Network, S, A, Y ):
        rho   = Network ( S, True )
        return tf.reduce_sum( tf.square( Y - ( rho * A ) ) ) / Beta
    
    
    def start_training ( self, Symbol,  Network, Network_T,  W, H, f_idx, parameters, TopN  ):

        # create directories
        if os.path.isdir ( './temp/') == False:
            os.mkdir( './temp/')
        if os.path.isdir ( './'+ Symbol +'/') == False:
            os.mkdir( './'+ Symbol +'/')

        if os.path.isdir ( './'+ Symbol + '/weights/' ) == False:
            os.mkdir('./'+ Symbol + '/weights/' )

        #################################################
        # M, B, C, Beta, Learning_rate, P 
        M       = int( parameters[0] )
        B       = int( parameters[1] )
        C       = int( parameters[2] )
        Beta    = int (parameters[3] )

        LRate   = parameters[4]
        P       = parameters[5]
        ################################################
        LossR   = 0                        # Loss
        memory  = exR.exRep( M, W, H )     # set memory buffer for experience replay
        
        opt     = tf.train.AdamOptimizer( LRate )   # adam optimizer

        preS    = tf.constant( 0, dtype =tf.float32, shape =[W,H] ) 
        preA    = tf.constant( 0, dtype =tf.int32, shape =[ self.num_actions ] ) 

        curS    = tf.constant( 0, dtype =tf.float32, shape =[W,H] ) 
        curA    = tf.constant( 0, dtype =tf.int32, shape =[ self.num_actions ] ) 

        curR    = 0
        nxtS    = tf.constant( 0, dtype =tf.float32, shape =[W,H] ) 
       

        listof_result   = list()    
        for b in range ( self.maxiter ) :

            #1.0 get random company index c, time index t
            c       = random.randrange( 0, len( self.TrX ) )
            t       = random.randrange( 1, len( self.TrX[c] ) -1  )

            #1.1 get previous state, action using epsilon greedy policy 
            preS    = tf.convert_to_tensor( self.TrX[c][t-1], dtype= tf.float32 )
            if( self.randf(0,1) <= self.epsilon):
                preA        = tf.convert_to_tensor  (  self.get_randaction   ( self.num_actions ) )
            else:                    
                rho         = Network               ( preS, False )
                eta         = tf.one_hot            ( tf.argmax( rho, 1), self.num_actions, dtype= tf.int32 )
                preA        = tf.reshape            ( eta, [self.num_actions] )
    
            #1.2 get current state, action using epsilon greedy policy 
            curS    = tf.convert_to_tensor( self.TrX[c][t], dtype= tf.float32 )
            if( self.randf(0,1) <= self.epsilon):
                curA        = tf.convert_to_tensor  ( self.get_randaction   ( self.num_actions ) )
            else:          
                rho         = Network               ( curS, False  )
                eta         = tf.one_hot            ( tf.argmax( rho, 1), self.num_actions, dtype= tf.int32 )
                curA        = tf.reshape            ( eta, [self.num_actions] )

            #1.3 get immediate reward and next state
            curR    = self.get_reward               ( preA, curA, self.TrY[c][t], P )
            nxtS    = tf.convert_to_tensor          ( self.TrX[c][t+1], dtype= tf.float32 )


            #1.4 store experience in the memory buffer: tuple of curS, curA, curR, nxtS   
            memory.remember( curS, curA, curR, nxtS )

            #1.5: set epsilon                       
            if ( self.epsilon > self.epsilon_min ):
                self.epsilon = self.epsilon * self.epsilon_decay

                
            #2: at every B iterations, update network parameter theta
            if ( len( memory.curS ) >= M ) and( b % B == 0 )  :
                
                #2.0: sample Beta size batch from memory buffer and take gradient step with repect to network parameter theta 
                S,A,Y   =  memory.get_Batch     ( Network_T, Beta, self.num_actions, self.Gamma  )
                S       = tf.convert_to_tensor  ( S, dtype= tf.float32 )
                A       = tf.convert_to_tensor  ( A, dtype= tf.float32 )
                Y       = tf.convert_to_tensor  ( Y, dtype= tf.float32 )
                
                #2.1: take gradient step to minimize LossR with repect to network parameter theta  
                with tf.GradientTape() as tape:
                    LossR   = self.get_LossR    ( Beta, Network, S, A, Y )

                grads   = tape.gradient         ( LossR, Network.trainable_weights )
                opt.apply_gradients             ( zip ( grads, Network.trainable_weights ) )

                #2.2:  update target network parameter theta*
                if( b % ( C * B ) == 0 )  : 
                    Network.save_weights     ( './para/Network' )
                    Network_T.load_weights   ( './para/Network' )
                    
                #2.3: evaluate the performance of the network parameterized by theta, at the validation set and save parameters theta
                if(( b % (  30 * C * B ) == 0 ) and ( b > 0 )) :
                    result  =  self.evaluate_network ( self.ValX, self.ValY, Network,  self.num_actions, W, H, b )
            
                    print 'Iter: ', b
                    # write, save, 
                    listof_result.append    ( result )
                    self.writeResult        ( './' + Symbol + '/ValidationResult.csv' , result )     
                    Network.save_weights    ( './temp/' + self.get_filename(f_idx, b))
                    self.copytree           ( './temp/' + self.get_filename(f_idx, b), './' + Symbol+ '/weights/' +  self.get_filename(f_idx, b) )

        # get the best performing parameters in validation set
        listof_result = sorted  ( listof_result, key=itemgetter(0), reverse=True )
        self.delete_files       ( Symbol, listof_result, f_idx, TopN )

        # delete temporary file
        shutil.rmtree           ( './para/' , ignore_errors = True)
        return 0 

    def evaluate_network            ( self, DataX, DataY,  Network, num_actions, W, H, b  ):
       
        # list
        N           = len( DataX    )
        Days        = len( DataX[0] )               
        curA        = np.zeros((N, num_actions ))
        inputs_t    = np.zeros((N, W, H ))

        # portfolio at previous time step t-1 and current time step t 
        preAlpha_n  = np.zeros( N )
        curAlpha_n  = np.zeros( N )
        posChange   = 0

        # reward
        curR        = np.zeros( N )
        avgDailyR   = np.zeros( Days )

        # total return, Long, Neutral, Short 
        return_sum  = 0
        LS          = np.zeros(2, dtype=float ) 

        # evaluate the performance of current network
        for t in range ( Days - 1 ):
    
            # 1.0 get current input at time t
            for c in range ( N ):
                inputs_t[c] = DataX[c][t]

            # 1.1 get rho and choose action given current input 
            rho          = Network ( tf.convert_to_tensor( inputs_t , dtype=tf.float32), False )
            for c in range( N ):
                curA[c] = tf.one_hot ( tf.argmax( rho[c], 0), num_actions, dtype= tf.int32 )

            # 1.2 set portfolio based on current action at time t
            curAlpha_n  = self.get_Portfolio ( curA,  N  )

            # 1.3 evaluate rewards
            for c in range ( N ) :

                #1: get daily rewards sum :
                curR[c]                     = np.round(  curAlpha_n[c] * DataY[c][t], 8)
                avgDailyR[t]                = np.round(  avgDailyR[t] + curR[c], 8 )

                #2: get pos change sum
                posChange                   = np.round(  posChange +  abs( curAlpha_n[c] - preAlpha_n[c] ), 8)
                preAlpha_n[c]               = curAlpha_n[c]

                #3: get ratio of long and short position
                if curAlpha_n[c] > 0 :
                    LS[0] = LS[0] + abs( curAlpha_n[c] ) 
                else:
                    LS[1] = LS[1] + abs( curAlpha_n[c] ) 

            # 1.4 aggregate reward
            return_sum = round( return_sum + avgDailyR[t] , 4 )
       
        # result
        result = np.zeros( (6) )

        # return sum, annualized return
        result[0]   = return_sum
        result[1]   = round( return_sum * 252.0 / (Days - 1), 4)

        # position change, annualized position change
        result[2] = round( posChange, 4)
        result[3] = round( posChange * 252.0 / (Days - 1 ), 4)

        # calculate the ratio of long, short position
        result[4] = round( float(LS[0]) / float (LS[0] + LS[1]) , 4)

        # current iteration number
        result[5] = b 

	return result


    def randf           ( self,  s, e):
        return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

    def get_randaction  ( self,  numofactions ) :
        actvec      =  np.zeros ( (numofactions), dtype = np.int32 )  
        idx         =  random.randrange(0,numofactions)
        actvec[idx] = 1
        return actvec

    def get_Portfolio        ( self, curA, N ):         
        alpha       = np.zeros( N )
        
        # get average
        for c in range ( N ):
            alpha[c]    = 1 - np.argmax( curA[c] )

        #set alpha
        sum_a       = 0
        for c in range ( N ):
            sum_a   = np.round( sum_a + abs(alpha[c]), 4 )

        # set alpha
        if sum_a == 0 :
            return alpha

        for c in range ( N ):
            alpha[c] =np.round(  alpha[c] / sum_a, 8 )

        return alpha

    def get_reward              ( self, preA, curA, inputY, P ):
        pre_act = 1 - np.argmax( preA )
        cur_act = 1 - np.argmax( curA )
        return (cur_act * inputY) - P * abs( cur_act - pre_act )

    def get_filename            ( self, f, b ):
        return 'IDX_' +  str(f) + '_' + str(b) + '/'

    def copytree                ( self, src, dst, symlinks=False, ignore=None):
        if os.path.isdir( dst ) == False:
            os.mkdir( dst )

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def delete_files                ( self, Symbol, resultList, f, TopN ):
        shutil.rmtree( './temp/', ignore_errors = True )
        if len( resultList ) <= TopN :
            return 0
        b = 0
        for i in range( TopN, len( resultList ) ):
            b = int( resultList[i][-1] )
            shutil.rmtree  ( './' +Symbol+ '/weights/' + self.get_filename(f,b), ignore_errors = True)
        return 0

    def writeResult    ( self, filename, curResult ):
        f = open( filename, 'a' )
        
        #if isFirst == True:
         #   f.write('return_sum,return_annual,pos_change_sum,pos_change_annual,longshort_ratio,num_iterations,')
          #  f.write("\n")

        for i in range ( len( curResult ) ):
            f.write( str(curResult[i]) + ',' )
        f.write("\n")
        f.close()


