import tensorflow as tf
import numpy as np
import test  as  TS
import DataPPRL as DRL

import trainingSL as  TR
import CNN_SL as oC
import MLP_SL as oM

class startExperiment:

    def __init__ ( self, a ):
        self.DRead          = DRL.DataReaderRL()

	self.rootTR         = '../Data/Training/SL/'
	self.rootVL         = '../Data/Validation/'
	self.rootTS         = '../Data/Test/'

        self.num_actions    = 3
  
    def SL_MLP          ( self, TopN,  maxiter, paralistSL, Symbol, NetworkType, PorPV, TRCost,  Tau_w, Tau_s  ):
    
        ### GET TRAINING, VALIDATION, and TEST DATA
	pathsTRVL, pathsTEST = self.getData ( NetworkType, PorPV )

        # length of input vector
        if PorPV == 'VP':
            W   = 32               
        elif PorPV == 'VPV':
            W   = 64
        H   = 1          

	### START TRAINING ###  
	#################################################################################
	for f in range ( len ( pathsTRVL ) ):
	    XTr,YTr, XVal,YVal  = self.DRead.setData_TRVL   ( pathsTRVL[f] )
	    trm                 = TR.trainModel             ( self.num_actions, maxiter  )
	    trm.set_Data( XTr, YTr, XVal, YVal )

	    # MLP
	    Net                 = oM.constructDense ( self.num_actions, W )

	    # start training 
	    trm.start_training( Symbol, Net,  W, H, f, paralistSL, TopN  ) 

        ### START TESTING ###
	##################################################################################
	tsm         = TS.testModel( self.num_actions )
	WtList      = tsm.get_foldername_list ( './'+ Symbol + '/weights/' )

	initAsset   = 1.0 
	for f  in range ( len (pathsTEST ) ):
	    XVal, XTest,YTest,WTest = self.DRead.setData_Test   ( pathsTEST[f] )
	    tsm                     = TS.testModel              ( self.num_actions ) 
	    tsm.set_Data ( XVal, XTest, YTest, WTest )

	    Net                     = oM.constructDense         ( self.num_actions, W )
	    initAsset               = tsm.start_testing         ( Symbol, Net, W, H, f, WtList[f], Tau_w, Tau_s, initAsset, (TRCost / 2.0) )

    def SL_CNN          ( self, TopN,  maxiter, paralistSL, Symbol, NetworkType, PorPV, TRCost,  Tau_w, Tau_s  ):
    
        ### GET TRAINING, VALIDATION, and TEST DATA
	pathsTRVL, pathsTEST = self.getData ( NetworkType, PorPV )
        W   = 32               
        H   = 32          

	### START TRAINING ###  
	#################################################################################
	for f in range ( len ( pathsTRVL ) ):
	    XTr,YTr, XVal,YVal  = self.DRead.setData_TRVL   ( pathsTRVL[f] )
	    trm                 = TR.trainModel             ( self.num_actions, maxiter  )
	    trm.set_Data( XTr, YTr, XVal, YVal )

	    # MLP
	    Net                 = oC.constructCNN ( self.num_actions, W )

	    # start training 
	    trm.start_training( Symbol, Net,  W, H, f, paralistSL, TopN  ) 

        ### START TESTING ###
	##################################################################################
	tsm         = TS.testModel( self.num_actions )
	WtList      = tsm.get_foldername_list ( './'+ Symbol + '/weights/' )

	initAsset   = 1.0 
	for f  in range ( len (pathsTEST ) ):
	    XVal, XTest,YTest,WTest = self.DRead.setData_Test   ( pathsTEST[f] )
	    tsm                     = TS.testModel              ( self.num_actions ) 
	    tsm.set_Data ( XVal, XTest, YTest, WTest )

	    Net                     = oC.constructCNN           ( self.num_actions, W )
	    initAsset               = tsm.start_testing         ( Symbol, Net, W, H, f, WtList[f], Tau_w, Tau_s, initAsset, (TRCost / 2.0) )

    def getData         ( self, NetworkType, PorPV ):
        ## NetworkType      ## PorPV
        ## MLP              ## VP       ( vector of prices )
        ## MLP              ## VPV      ( vector prices and  volumes )

        ## CNN              ## CP       ( chart consist of prices )
        ## CNN              ## CPV      ( chart consist of prices and volumes )

        # Training, Validation
	pathsTRVL = list()
	pathsTRVL.append(  self.DRead.get_paths_TRVL (
	    self.rootTR + 'IND_' + PorPV + '_IND_W32_2000_2004/', 
	    self.rootVL + 'IND_' + PorPV + '_IND_W32_2004_2006/',  NetworkType ) )
        
        pathsTRVL.append(  self.DRead.get_paths_TRVL (
	    self.rootTR + 'IND_' + PorPV + '_IND_W32_2004_2008/', 
	    self.rootVL + 'IND_' + PorPV + '_IND_W32_2008_2010/',  NetworkType ) )

        pathsTRVL.append(  self.DRead.get_paths_TRVL (
	    self.rootTR + 'IND_' + PorPV + '_IND_W32_2008_2012/', 
	    self.rootVL + 'IND_' + PorPV + '_IND_W32_2012_2014/',  NetworkType ) )

        # Validation, Test
	pathsTEST = list()
	pathsTEST.append(  self.DRead.get_paths_Test (
	    self.rootVL + 'IND_' + PorPV + '_IND_W32_2004_2006/',
	    self.rootTS + 'IND_' + PorPV + '_SNP_W32_2006_2010/', NetworkType ) )
	
        pathsTEST.append(  self.DRead.get_paths_Test (
	    self.rootVL + 'IND_' + PorPV + '_IND_W32_2008_2010/',
	    self.rootTS + 'IND_' + PorPV + '_SNP_W32_2010_2014/', NetworkType ) )

        pathsTEST.append(  self.DRead.get_paths_Test (
	    self.rootVL + 'IND_' + PorPV + '_IND_W32_2012_2014/',
	    self.rootTS + 'IND_' + PorPV + '_SNP_W32_2014_2018/', NetworkType ) )

        return pathsTRVL, pathsTEST






















