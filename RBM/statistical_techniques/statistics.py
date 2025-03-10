# Packes
from numpy.random import randint, uniform
import numpy as np
from time import time
from numpy.linalg import inv

'''
===============================================================================
------------------------------
Class statistical techniques :
------------------------------

To set the stadistic technique for better statistic analysis
===============================================================================
'''
class StatisticalTechniques:
    def __init__(self) -> None:
        pass
        
    # Returns mean of bootstrap samples                                                                                                                                                
    def statistic_apply(self,Data):
        return np.mean(Data)
    
    '''
    ===============================================================================
    ---------------------
    Bootstrap technique :
    ---------------------

    It needs to specify the number the resampling (Number_resampling) and 
    data (Data) where it is will be apply the technique.
    ===============================================================================
    '''
    # Bootstrap algorithm                                                                                                                                                              
    def bootstrap(self,
        Data,                       # Data to do bootstrap technique
        Number_resampling: int,          # NUmber the resampling to made a bootstrap technique 
        Applay_statistics = None,   # Information you want to know about the data (Normally will be mean and variance value)
        ):
        if Applay_statistics == None :
            Applay_statistics = self.statistic_apply

        else:
            print('< applay_statistic > : You have applied your own statistic')
        
        t0 = time()                                         # Initialize time
        Save_data_statidtic = np.zeros(Number_resampling);    # Save data for each resampling
        NData = len(Data);                                  # Number of data points

        # Non-parametric bootstrap                                                                                                                                                     
        for Nrs in range(Number_resampling):
            Save_data_statidtic[Nrs] = Applay_statistics(Data[randint(0,NData,NData)])  # Calculate the statistic for each resampling

        # Analysis result                                                                                                                                                                   
        print("Time consuming: %g sec" % (time()-t0)); 

        print("Original: ")
        print("Mean value  " , "  Variance", "  Error")
        print("%12g %15g %15g" % (np.mean(Data), np.std(Data), np.sqrt(np.std(Data/NData))))

        print("After Bootstrap technique: ")
        print("Mean value  " , "  Variance", "  Error")
        print("%12g %15g %15g" % (np.mean(Save_data_statidtic),np.std(Save_data_statidtic), np.sqrt(np.std(Save_data_statidtic/Number_resampling))))

        return  Save_data_statidtic
    
    '''
    ===============================================================================
    --------------------
    Blocking technique :
    --------------------

    It needs to specify the data where it is will be apply the technique(Data),
    the number of data has to be a power of two (2^n) 
    ===============================================================================
    '''    
    # Blocking technique
    def blocking(self,
                 Data
                 ):
        
        # Start time
        t0 = time() 

        # Data input
        Data_input = Data

        # Preliminaries
        d = np.log2(len(Data))
        if (d - np.floor(d) != 0):
            print("Warning: Data size = %g, is not a power of 2." % np.floor(2**d))
            print("Truncating data to %g." % 2**np.floor(d) )
            Data = Data[:2**int(np.floor(d))]
        d = int(np.floor(d))
        n = 2**d
        s, gamma = np.zeros(d), np.zeros(d)
        mu = np.mean(Data)

        # Estimate the auto-covariance and variances for each blocking transformation
        for i in np.arange(0,d):

            # Data numbers
            n = len(Data)

            # Estimate autocovariance of Data
            gamma[i] = (n)**(-1)*sum( (Data[0:(n-1)]-mu)*(Data[1:n]-mu) )

            # Estimate variance of Data
            s[i] = np.var(Data)

            # perform blocking transformation
            Data = 0.5*(Data[0::2] + Data[1::2])
    
        # generate the test observator M_k from the theorem
        M = (np.cumsum( ((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1] )  )[::-1]

        # we need a list of magic numbers
        q =np.array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272, 
                  16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
                  24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 
                  31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
                  38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 
                  45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

        # use magic to determine when we should have stopped blocking
        for k in np.arange(0,d):
            if(M[k] < q[k]):
                break
        if (k >= d-1):
            print("Warning: Use more data")
        
        # Mean value 
        mean = mu

        # Variance 
        varince =   s[k]/2**(d-k)

        # Error
        error = np.sqrt(varince)

        # Show statistic result                                                                                                                                                                  
        print("Time consuming: %g sec" % (time()-t0)); 

        print("Original: ")
        print("Mean value  " , "  Variance", "  Error")
        print("%12g %15g %15g" % (np.mean(Data_input), np.std(Data_input), np.sqrt(np.std(Data_input))))

        print("After Blocking technique: ")
        print("Mean value  " , "  Variance", "  Error")
        print("%12g %15g %15g" % (mean, varince, error))
        return mean, varince, error
    
if __name__ == "__main__":
    '''
    Simulate the gropund state of a bosson in the spheric trap
    '''
    print('===========================================')
    print('THIS IS A TEST, IT HAS NOT PHYSICS MEANING')
    print('===========================================')
    Data = uniform(0.49,0.51,2**8)
    print('================= Bloking =================')
    print(StatisticalTechniques().blocking(Data))
    print('===========================================')
    print('================ Bootstrap ================')
    print(StatisticalTechniques().bootstrap(Data,Number_resampling=2**8))
    print('===========================================')