import numpy as np
import os


class DataReaderRL:

    def __init__ ( self ):
        self.a = 0

    def get_paths_TRVL   ( self, folderT, folderV, dtype ):

        paths = list()

        paths.append ( folderT + 'inputX.txt' )
        paths.append ( folderT + 'inputY.txt' )

        paths.append ( folderV + 'inputX.txt' )
        paths.append ( folderV + 'inputY.txt' )

        paths.append ( dtype )
        return paths


    def get_paths_Test   ( self, folderV, folderT, dtype ):

        paths = list()

        paths.append ( folderV + 'inputX.txt' )
        paths.append ( folderV + 'inputY.txt' )
        
        paths.append ( folderT + 'inputX.txt' )
        paths.append ( folderT + 'inputY.txt' )
        paths.append ( folderT + 'Weights.txt' )

        paths.append ( dtype )
        return paths

    def setData_TRVL         ( self, paths ):
        
        XTr = list()
        YTr = list()

        XVal  = list()
        YVal  = list()

        if( paths[-1] == 'CNN' ):
            XTr     = self.readX_Chart    ( paths[0] )
            YTr     = self.readY_IND      ( paths[1], len(XTr), len(XTr[0]) )
            XVal    = self.readX_Chart    ( paths[2] )
            YVal    = self.readY_IND      ( paths[3], len(XVal), len(XVal[0]) )

        elif( paths[-1] == 'MLP' ):
            XTr     = self.readX_Vector   ( paths[0] )
            YTr     = self.readY_IND      ( paths[1], len(XTr), len(XTr[0]) )
            XVal    = self.readX_Vector   ( paths[2] )
            YVal    = self.readY_IND      ( paths[3], len(XVal), len(XVal[0]) )

        else:
            print 'wrong data type'
        return XTr, YTr, XVal, YVal


    def setData_Test         ( self, paths ):
        XVal = list()
        
        XTest = list()
        YTest = list()
        WTest = list()

        if ( paths[-1] == 'CNN' ):
            XVal    = self.readX_Chart  ( paths[0] )

            XTest   = self.readX_Chart    ( paths[2] )
            YTest   = self.readY_SNP      ( paths[3], len(XTest[0]) )
            WTest   = self.readWeight     ( paths[4], len(XTest), len(XTest[0]) )

        elif ( paths[-1] == 'MLP' ):
            XVal    = self.readX_Vector ( paths[0] )

            XTest   = self.readX_Vector   ( paths[2] )
            YTest   = self.readY_SNP      ( paths[3], len(XTest[0]) )
            WTest   = self.readWeight     ( paths[4], len(XTest), len(XTest[0]) )

        else:
            print 'wrong data type'

        return XVal, XTest, YTest, WTest


  
    def readX_Vector  (self, filepath ):

        # Generate height by wdith   input chart image
        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.strip('\nF\n').split( '\nF\n'  )
        DataX   =   list()

        for c in range( len(rawdata)  ) :
            # Days
            vects_seq  = rawdata[c].strip('\nE\n').split( '\nE\n' )
            H   =  len( vects_seq[0].strip('\n').split('\n') )
            W   =  len( vects_seq[0].strip('\n').split('\n')[0].strip(',').split(',') )

            # vects, FeatVects:  vectors of  company c,  day t
            FeatVects_seq = list()
            for t in range ( len(vects_seq) ):
                vects       = vects_seq[t].strip('\n').split('\n')
                FeatVects   = np.zeros( (H, W) )

                # spliting and converting types 
                for v in range ( H ):
                    values = vects[v].strip(',').split(',') 
                    for w in range ( W ) :
                        FeatVects[v][w] = float ( values[w] )
                
                FeatVects_seq.append(  FeatVects.flatten() )

            # Flatten FeatVects and add
            DataX.append ( FeatVects_seq )

        # flatten
        D0= np.shape( DataX )[0]
        D1= np.shape( DataX )[1]
        D2= np.shape( DataX )[2]
        DataX = np.reshape( DataX, (D0,D1,D2,1) )

        print 'INPUT:  Comp#, Days# VectorShape: ', len(DataX), len(vects_seq), np.shape( DataX[0][0] ), '   input shape ', np.shape( DataX )
        return DataX 

    def readX_Chart  (self, filepath):

        # Generate height by wdith   input chart image
        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.strip('\nF\n').split( '\nF\n'  )
        DataX   =   list()

        for c in range( len(rawdata) ) :
            state_seq  = rawdata[c].strip('\nE\n').split( '\nE\n' )

            # get H, W 
            H   =  len( state_seq[0].strip('\n').split('\n') )
            W   =  len( state_seq[0].strip('\n').split('\n')[0].strip(' ').split(' ') )
            # matrix seq for company c
            matrix_seq = list()
            for t in range ( len(state_seq) ):
                rows    = state_seq[t].strip('\n').split('\n')
                matrix  = np.zeros( ( H,W ) )

                # input matrix on day t
                for r in range ( H ):
                    row  = rows[r].split( ' ' )
                    for w in range( W ):
                        matrix[r][w] = int( row[w] )
                matrix_seq.append( matrix )
            DataX.append ( matrix_seq )

        print 'INPUT:  Comp#, Days# ChartShape: ', len(DataX), len(state_seq), np.shape( DataX[0][0] ) , '   input shape', np.shape( DataX ) 
        return DataX 


    def readWeight    ( self, filepath, N, Days ):
        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.split( '\n'  )
        DataW   = list()

        if ( len(rawdata)-1) != (N*Days) :
            print 'number of input data is invalid'

        cnt     = 0
        for c in range ( N ) :
            return_seq = list()

            for t in range ( Days)  :
                return_seq.append( float( rawdata [cnt] ) )
                cnt = cnt + 1

            DataW.append ( return_seq )

        print 'OUTPUT: Weight:',  np.shape( DataW )  
        return DataW 

                                          
    def readY_IND    ( self, filepath, N, Days ):
        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.split( '\n'  )
        DataY   = list()

        if ( len(rawdata)-1) != (N*Days) :
            print 'number of input data is invalid', len(rawdata)-1, N, Days

        cnt     = 0
        for c in range ( N ) :
            return_seq = list()

            for t in range ( Days)  :
                return_seq.append( float( rawdata [cnt] ) )
                cnt = cnt + 1

            DataY.append ( return_seq )

        print 'OUTPUT: IIorSS  Y: ' , np.shape( DataY )
        return DataY 

                                          
    def readY_SNP    ( self, filepath, Days ):
        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.strip('\n').split( '\n'  )
        DataY   = list()
        Y_seq   = list()

        if Days != len(rawdata) :
            print 'invalid lenth: ', Days, len(rawdata)

        cnt     = 0
        for t in range ( Days ):
            Y_seq.append( float( rawdata [cnt] ) )
            cnt = cnt + 1

        DataY.append ( Y_seq )

        print 'OUTPUT: IS Y:' , np.shape(DataY)
        return DataY 






