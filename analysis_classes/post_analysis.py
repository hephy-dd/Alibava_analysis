#this script is called from AliSys.py for each set of ped,cal,run files

import pdb
import copy

def main(ped, cal, run, results):
    #This is executed for each set of files
    myresults=copy.copy(results)#create copy of results dict
    print('### Scribbling Pad ###')
    print('Keys in \'results\'',results.keys())
    for i in [ped, cal, run]: print(i)
    
    ## put some code here

    #pdb.set_trace()
    
    print('##########################')
    return myresults

def final(myresults):
    #This is executed at the end of the analysis
    
    print(myresults)
    return 0
