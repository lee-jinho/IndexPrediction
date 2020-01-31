import tensorflow as tf
import numpy as np
import random
import datetime 
import exReplay as exR
import os
import shutil

class testModel:

    def __init__   ( self,  num_actions  ):
        self.num_actions    = num_actions

    def set_Data    ( self, ValX, TestX, TestY, TestW ):
 
        self.ValX   = ValX
        
        self.TestX   = TestX
        self.TestY   = TestY
        self.TestW   = TestW


        print 'Val X        :  Comp#, Days# ', len( self.ValX ), len( self.ValX[0] ), np.shape(  self.ValX )

        print 'Test X       :  Comp#, Days# ', len( self.TestX ), len( self.TestX[0] ), np.shape( self.TestX )
        print 'Test Y       :  Comp#, Days# ', len( self.TestY ), len( self.TestY[0] ), np.shape( self.TestY )
        print 'Test Weight  :  Comp#, Days# ', len( self.TestW ), len( self.TestW[0] ), np.shape( self.TestW )

    def get_foldername_list  ( self, folderpath ):
        paths = os.listdir( folderpath )            
        paths = sorted( paths )

        pathlist = list()
        for i in range ( len(paths) ):
            pathlist.append( folderpath + paths[i] + '/' )

        print pathlist
        return pathlist

    def start_testing       ( self, Symbol, Net, W, H, f, curPaths, Tau_w, Tau_s, initAsset, PosChangeCost  ):

        # load weights
        Net.load_weights( curPaths )

        # get mu_eta, sigma_eta from validation set 
        mu, sigma  = self.get_mu_sigma ( self.ValX, Net, W, H  )
        print 'mu, sigma: ',mu, sigma 

        # evaluate network
        resultT, assetList   = self.evaluate_network ( self.TestX, self.TestY, self.TestW, Net, W,
                H, Tau_w, Tau_s, mu, sigma, initAsset, PosChangeCost )

        # save results
        self.write_Result   ( './' + Symbol + '/TestResult.csv', f, resultT )
        self.write_Asset    ( './' + Symbol + '/AssetList.csv' , assetList )

        # return final asset
        return resultT[0]

    def get_mu_sigma ( self, ValX, Net, W, H  ):
        # list
        N           = len( ValX    )
        Days        = len( ValX[0] )               
        inputs_t    = np.zeros((N, W, H ))

        # 
        rho         = np.zeros(( N, 3))
        curLongQ    = np.zeros( N )   # LongSoftmax - ShortSoftmax

        eta         = 0 
        etaList     = np.zeros( Days -1 )

        for t in range ( Days - 1 ):
    
            # gather
            for c in range ( N ):
                inputs_t[c] = ValX[c][t]

            rho          = Net ( tf.convert_to_tensor( inputs_t , dtype=tf.float32), False )            
            for c in range ( N ):
                curLongQ[c] = rho[c][0] - rho[c][2]

            eta        = self.get_equal_weighted_eta    ( curLongQ, t )
            etaList[t] = eta 

        #
        mu      = np.mean( etaList ) 
        sigma   = np.std ( etaList )
	return  mu, sigma


    def evaluate_network ( self, DataX, DataY, Cap,  Network, W, H, Tau_w, Tau_s, mu, sigma,  initAsset, posChangeCost   ):

        # list
        N           = len( DataX    )
        Days        = len( DataX[0] )               
        inputs_t    = np.zeros((N, W, H ))

        # Alpha: 1, 0, or -1
        preAlpha    = 0 
        curAlpha    = 0
        posChange   = 0

        # reward
        curR        = 0
        avgDailyR   = np.zeros( Days )

        # cumulative asset:  
        cumAsset    = initAsset 
        return_sum  = 0
        LNS         = np.zeros(3) 

        # Subtracted value of rho[Long] - rho[Short] represecting how company i will sharply rise at
        # next timestep
        curLongQ    = np.zeros( N )   

        # weighted sum of subtracted value
        eta         = 0

        #for further analyze
        assetList   = np.zeros( Days )

        for t in range ( Days - 1 ):
    
            # 1.0 get current input at time t
            for c in range ( N ):
                inputs_t[c] = DataX[c][t]

            # 1.1 get rho and 
            rho          = Network ( tf.convert_to_tensor( inputs_t , dtype=tf.float32), False )
            for c in range ( N ):
                curLongQ[c] = rho[c][0] - rho[c][2]

            # 1.2 calculate eta 
            eta             = self.geteta               ( curLongQ, Cap, t )

            # 1.3 decide long, neutral, or short the S&P500
            curAlpha        = self.laggedPosChange      ( eta, preAlpha, mu, sigma, Tau_w, Tau_s   )

            # 1.4 get daily reward 
            curR            = self.get_reward           ( preAlpha, curAlpha, DataY[0][t], posChangeCost )
            avgDailyR[t]    = np.round                  (  avgDailyR[t] + curR, 8 )

            # 1.5 aggregate reward
            return_sum      = np.round                  (  avgDailyR[t] + return_sum , 8 )

            # 1.6: get pos change sum
            posChange       = np.round(  posChange +  abs( curAlpha - preAlpha ), 8)
            preAlpha        = curAlpha
    
            # 1.7: Long Short
            if curAlpha > 0 :
                LNS[0] = LNS[0] + 1
            elif curAlpha < 0:
                LNS[2] = LNS[2] + 1
            else:
                LNS[1] = LNS[1] + 1

        # calculate MDD: Maximum Drawdown
        MDD = self.getDrawDown ( avgDailyR, Days-1 )

        # calculate cumulative return
        for t in range( Days ):
            cumAsset        = round ( cumAsset + ( cumAsset * avgDailyR[t] * 0.01  ), 8 )
            assetList[t]    = cumAsset

        # return
        result = np.zeros( (8) )

        # cummulative asset
        result[0] = cumAsset       

        # return sum, annualized return
        result[1] = return_sum
        result[2] = round( return_sum * 252.0 / (Days - 1), 4)

        # position change, annualized position change
        result[3] = posChange
        result[4] = round(  posChange * 252.0 / (Days - 1 ), 4)

        # ratio of Long, Short position 
        result[5] = round( float(LNS[0]) / float (LNS[0] + LNS[2]) , 4)

        # NNP: ratio of non-neutral position 
        result[6] = round( float(LNS[0] + LNS[2] ) / float( LNS[0] + LNS[1] + LNS[2] ) , 4 )

        # MDD:  Maximum Drawdown
        result[7] = round(  MDD , 4 )
	return result, assetList


    def laggedPosChange        ( self, eta, preAlpha, mu, sigma, Tau_w, Tau_s  ):
        UprTH_s   = np.round( mu +  ( sigma * Tau_s ), 4 )
        UprTH_w   = np.round( mu +  ( sigma * Tau_w ), 4 )

        LwrTH_s   = np.round( mu -  ( sigma * Tau_s ), 4 )
        LwrTH_w   = np.round( mu -  ( sigma * Tau_w ), 4 )

        if preAlpha == 0:
            if eta > UprTH_s:
                return 1
            elif eta < LwrTH_s:
                return -1
            else:
                return 0

        elif preAlpha == 1:
            if eta > UprTH_w:
                return 1
            elif eta < LwrTH_s:
                return -1
            else:
                return 0

        elif preAlpha == -1:
            if eta < LwrTH_w:
                return -1
            elif eta > UprTH_s:
                return 1
            else:
                return 0
        else:
            print 'Value of preAlpha is wrong:  ', preAlpha

        return 0


    def get_equal_weighted_eta       ( self, LongQ, t  ):
        Qsum =0
        for i in range ( len(LongQ) ):
            Qsum    =   Qsum +  LongQ[i];
        return Qsum / len(LongQ)


    def geteta       ( self, LongQ, DataW, t  ):
        Qsum =0
        DWSum =0  
        for i in range ( len(LongQ) ):
            Qsum    =   Qsum + ( DataW[i][t] * LongQ[i] )  ;
            DWSum   =   DWSum + DataW[i][t]

        return Qsum/ DWSum 
    
    def get_reward              ( self, preA, curA, inputY, P ):
        return np.round( (curA * inputY) - P * abs( curA - preA ), 8 )

    def getDrawDown             ( self, avgDailyR, Days ):
        cumAsset    = 1.0
        Max         = 1.0
        Min         = 1.0

        drawdownlist = list()
        for t in range ( Days ):
            cumAsset = np.round ( cumAsset + ( cumAsset * avgDailyR[t] * 0.01  ), 8 )
            if cumAsset > Max:
                Max = cumAsset
                Min = cumAsset
            else:
                if cumAsset < Min:
                    Min = cumAsset
                    drawdownlist.append( 100 * (Min - Max) / Max  )

        drawdownlist.sort()
        return drawdownlist[0] 

    def write_Asset  ( self, filename, cumAssetList ) :
        f = open( filename, 'a' )
        for i in range ( len( cumAssetList ) ):
            f.write( str( cumAssetList[i] ) + ',' )
            f.write("\n")
        f.close()

    def write_Result  ( self, filename, fidx, curResult ) :
        f = open( filename, 'a' )

        #f.write  ('fileIndex,cum_asset,return_sum,return_annual,pos_change_sum,pos_change_annual,longshort_ratio,NNP,MDD,')
        #f.write( '\n')

        f.write(  str(fidx) + ','   )
        for i in range ( len( curResult ) ):
            f.write( str( curResult[i] ) + ',' )

        f.write("\n")
        f.close()



