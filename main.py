import argparse
import subprocess
import os
parser = argparse.ArgumentParser(description='Running KGE benchmark libraries')

parser.add_argument('-lib','--library_name',help='Name of the library')
parser.add_argument('-hp','--hyperparameter_tuning',help="Where to run hyperparameter tuning of parameters")

args = parser.parse_args()


if __name__ == '__main__':
    if args.lib == 'ampligraph':
        from ampligraph import kgeCode, kgeCode_best
        os.chdir("ampligraph")
        if args.hp == 'true':
            command = ['python3','kgeCode.py']
        elif args.hp == 'false':
            command = ['python3','kgeCode_best.py']
    
    elif args.lib == 'openke':
        from ampligraph import kgeCode, kgeCode_best
        os.chdir("openKE")
        if args.hp == 'true':
            command = ['python3','kgeCode.py']
        elif args.hp == 'false':
            command = ['python3','kgeCode_best.py']
    
    elif args.lib == 'pykeen':
        from ampligraph import kgeCode, kgeCode_best
        os.chdir("pykeen")
        if args.hp == 'true':
            command = ['python3','kgeCode.py']
        elif args.hp == 'false':
            command = ['python3','kgeCode_best.py']
    
    elif args.lib == 'pytorch_geometric':
        from ampligraph import kgeCode, kgeCode_best
        os.chdir("pytorch_geometric")
        if args.hp == 'true':
            command = ['python3','kgeCode.py']
        elif args.hp == 'false':
            command = ['python3','kgeCode_best.py']
    
    elif args.lib == 'torchkge':
        from ampligraph import kgeCode, kgeCode_best
        os.chdir("TorchKGE")
        if args.hp == 'true':
            command = ['python3','kgeCode.py']
        elif args.hp == 'false':
            command = ['python3','kgeCode_best.py']
    
    elif args.lib == 'pykg2vec':
        from ampligraph import kgeCode, kgeCode_best
        os.chdir("pykg2vec_hpo")
        if args.hp == 'true':
            command = ['python3','kgeCode.py']
        elif args.hp == 'false':
            command = ['python3','kgeCode_best.py']