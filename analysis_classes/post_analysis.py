#this script is called from AliSys.py for each set of ped,cal,run files

import pdb

def main(ped, cal, run, results):
    #This is executed for each set of files
    print('### Scribbling Pad ###')
    print('Keys in \'results\'',results.keys())
    for i in [ped, cal, run]: print(i)

    ## put some code here
    results=[]
    pdb.set_trace()
    
    print('##########################')
    return results

def final(results):
    #This is executed at the end of the analysis
