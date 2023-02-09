import sys, glob, os
sys.path.append('deep-learning-base')

for p in glob.glob('deep-learning-base/*'):
    if os.path.isdir(p):
        sys.path.append(p)