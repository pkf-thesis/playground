import librosa
import numpy
from librosa.feature import mfcc

def loadFile(path):
    print('Loading %s' % path)

    y, sr = librosa.load(path)
    
    'Mel-frequency cepstral coefficients (MFCCs) - https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html'
    data = mfcc(y=y, sr=sr).T
    return data