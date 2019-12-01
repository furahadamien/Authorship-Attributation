	
import shutil

import os

for x in range(1, 2):
    temp = 'candidate000'+str(x)
    source = '../Copora - furaha/pan12-authorship-attribution-test-corpus-2012-05-24/'
    #source= '../Copora - furaha/Training/pan11-authorship-attribution-test-dataset-large-2015-10-20/'+temp+'/'
    dest1= '../Copora - furaha/Testing/unknown/'

    path = "../Copora - furaha/Testing/unknown/"

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    files = os.listdir(source)
    length = len(files)
    count = 0
    for f in files:
        #if count < len(files)/2:
        shutil.move(source+f, dest1+f)
        count = count +1



    
        