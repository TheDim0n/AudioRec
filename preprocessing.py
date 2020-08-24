import librosa as li
import numpy as np
import scipy as sp
import tensorflow as tf

class Audio():
    def __init__(self, path):
        self.sr = 44100
        self.path = path
        self.data = li.load(self.path, sr=self.sr)[0]
        self.e_parts = self.get_energy()
        self.label = int(self.path.split('\\')[-3].split()[0])
        
    def pitch_shift(self, y=[]):
        if len(y) == 0:
            y = self.data
        y1 = li.effects.pitch_shift(y, self.sr, n_steps=-2)
        y2 = li.effects.pitch_shift(y, self.sr, n_steps=2)
        return [y1, y2]
    
    def add_noise(self, y=[]):
        if len(y) == 0:
            y = self.data
        wn = np.random.random_sample(len(y))
        y_wn = y + 0.005*wn
        return [y_wn]
    
    def augmented(self):
        augments = [self.data]
        if self.label == 1:
            augments += self.add_noise()
            augments += self.pitch_shift(augments[0])
            augments += self.pitch_shift(augments[1])
        data = []
        for i in augments:
            data.append(self.get_energy(i))
        return data
        
    def get_energy(self, y=[]):
        if len(y) == 0:
            y = self.data
        x = tf.keras.preprocessing.sequence.pad_sequences(
            [y], 
            maxlen=int(self.sr * 0.4), 
            padding='post',
            truncating='post',
            dtype='float32'
        )[0]
        coeff = sp.signal.firwin(999, [260, 700], fs=self.sr, pass_zero=False)
        x_filtered = sp.signal.lfilter(coeff, 1.0, x)
        x_normalized = x_filtered/x_filtered.max()
        x_squared = np.square(x_normalized)
        splited = np.array_split(x_squared, 20)
        e_parts = np.empty((0))
        for part in splited:
            e_parts = np.append(e_parts, sp.integrate.simps(part))
        return e_parts

def get_MFCC(path, sr=44100):
    y = li.load(path, sr=sr)[0]
    s = li.feature.melspectrogram(y=y, sr=sr)
    log_s = li.power_to_db(s)
    features = li.feature.mfcc(y=y, sr=sr, S=log_s, n_mfcc=40)
    features = np.mean(features, axis=1)
    return features

def stretch(y):
    faster = li.effects.time_stretch(y, 1.1)
    slower = li.effects.time_stretch(y, 0.9)
    return [slower, faster]

def add_noise(y):
    wn = np.random.random_sample(len(y))
    y_wn = y + 0.005*wn
    return [y_wn]

def augment_data(y, sr):
    data = [y,]
    data += pitch_shift(y, sr)
    new = []
    for i in data:
        new += stretch(i)
    data += new
    new = []
    for i in data:
        new += add_noise(i)
    data += new
    return data

if __name__ == "__main__":
    pass
