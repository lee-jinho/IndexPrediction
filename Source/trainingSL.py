import tensorflow as tf
import numpy as np
import random
import shutil 
import os
from operator import itemgetter


class trainModel:

    def __init__   ( self,  num_actions,  maxiter  ):
        self.num_actions    = num_actions
        self.maxiter        = maxiter
        

    def set_Data    ( self, TrX, TrY, ValX, ValY ):
        self.TrX    = TrX
        self.TrYL   = np .zeros( ( len( TrY ), len ( TrY[0] ), self.num_actions ), dtype = float )
        
        self.ValX    = ValX
        self.ValY    = ValY

        # set label
        for c in range ( len( self.TrYL ) ):
            for t in range ( len( self.TrYL[c] ) ):
                if TrY[c][t] > 0 :
                    self.TrYL[c][t][0] = 1
                elif TrY[c][t] < 0 :
                    self.TrYL[c][t][2] = 1
                else:
                    self.TrYL[c][t][1] = 1

        print 'Training X   :  Comp#, Days# ', len( self.TrX ), len( self.TrX[0] ), np.shape( self.TrX )
        print 'Training Y   :  Comp#, Days# ', len( self.TrYL ), len( self.TrYL[0] ), np.shape( self.TrYL )
        print 'Validation X :  Comp#, Days# ', len( self.ValX ), len( self.ValX[0] ), np.shape( self.ValY )
        print 'Validation Y :  Comp#, Days# ', len( self.ValY ), len( self.ValY[0] ), np.shape( self.ValY )
    
    def start_training ( self, Symbol,  Network, W, H, f_idx, parameters, TopN  ):

        # create directories
        if os.path.isdir ( './temp/') == False:
            os.mkdir( './temp/')
        if os.path.isdir ( './'+ Symbol +'/') == False:
            os.mkdir( './'+ Symbol +'/')
        if os.path.isdir ( './'+ Symbol + '/weights/' ) == False:
            os.mkdir('./'+ Symbol + '/weights/' )

        #################################################
        # Beta, Learning_rate
        Beta    = int (parameters[0] )
        LRate   = parameters[1]
        ################################################
        LossS   = 0                                 # Loss        
        opt     = tf.train.AdamOptimizer( LRate )   # adam optimizer

        listof_result   = list()    
        for b in range ( self.maxiter ) :

        
            #1  get random index list
            IdxList         = self.set_IDXListsRand ( Beta, self.TrX )


            #2 get Batch
            X, Y            = self.get_Batch        ( IdxList, self.TrX, self.TrYL, W, H, self.num_actions )


            #3 get Loss
            with tf.GradientTape() as tape:
                LossS    = self.get_LossS         ( tf.convert_to_tensor( X, dtype=tf.float32),  
                        tf.convert_to_tensor(  Y, dtype = tf.float32 ), Network, Beta  )

            grads   = tape.gradient         ( LossS, Network.trainable_weights )
            opt.apply_gradients( zip ( grads, Network.trainable_weights ) )

            #4 validate && save
            if(( b % ( 20000 ) == 0 ) and ( b > 0 )) :
                result  =  self.evaluate_network ( self.ValX, self.ValY, Network, self.num_actions, W, H, b )

                print 'Iter: ' , b 

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

    def get_LossS        ( self, curIn, Y, Network, Beta  ):
        rho   = Network ( curIn, True )
        return  tf.reduce_sum( - Y * tf.log( rho ) ) / Beta

  
    def set_IDXListsRand      ( self, Beta, DataList ):
        IdxList = np.zeros ( (Beta,2), dtype= int )
        for i in range ( Beta ):
            IdxList[i,0] = random.randint( 0, len( DataList )-1 )
            IdxList[i,1] = random.randint( 0, len( DataList[0] )-1 )

        return IdxList

    def get_Batch           ( self, Idx, DataX, DataY, W, H, num_actions ):
        curS = np.zeros( (len(Idx), W, H) )
        curY = np.zeros( (len(Idx), num_actions ) )
        
        for i in range ( len(Idx) ):
            curS[i] = DataX[ Idx[i,0]][Idx[i,1] ]
            curY[i] = DataY[ Idx[i,0]][Idx[i,1] ]

        return curS, curY



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

    def writeResult    ( self, filename, curResult):
        f = open( filename, 'a' )

        for i in range ( len( curResult ) ):
            f.write( str(curResult[i]) + ',' )
        f.write("\n")
        f.close()


