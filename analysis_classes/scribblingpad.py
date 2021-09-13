#this script is called from AliSys.py for each set of ped,cal,run files

import pdb

def main(ped, cal, run, results):
    print('### Scribbling Pad ###')
    print('Keys in \'results\'',results.keys())
    for i in [ped, cal, run]: print(i)

    ## put some code here
    #pdb.set_trace()
    
    print('##########################')
