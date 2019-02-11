import mp3_npy
import numpy
import sys

filePath = str(sys.argv[1])
data = mp3_npy.loadFile(filePath)
print(data)

'''
Save to file:
numpy.save("NAME", data)
'''