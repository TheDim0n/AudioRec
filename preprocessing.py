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
        self.label = self.get_label()
        
    def get_label(self):
        label = int(self.path.split('\\')[-3].split()[0])
        if label > 0:
            label = 1
        return label

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
        y_wn = y + 0.005 * wn
        return [y_wn]

    def reverb(self, sound, start, shift, fade, times):
        result = np.copy(sound)
        reverbed = np.concatenate((np.zeros(start), result[:result.size - start] * fade))
        result = result + reverbed
        for i in range(times):
            reverbed = np.concatenate((np.zeros(shift), reverbed[:reverbed.size - shift] * fade))
            result = result + reverbed
        return result

    def echo(self, sound, sampleRate, shift, shiftProgress, fade, times):
        result = np.copy(sound)
        result = result + self.reverb(result, int(sampleRate * 0.045), int(sampleRate * 0.0001), 0.3, 50)
        currentEcho = np.copy(sound)
        for i in range(times):
            currentEcho = np.concatenate(
                (np.zeros(int(sampleRate * shift)), currentEcho[:currentEcho.size - int(sampleRate * shift)])) * fade
            shift *= shiftProgress
            tmp = self.reverb(currentEcho, int(sampleRate * 0.045), int(sampleRate * 0.0001), 0.3, 50)
            result = result + tmp
        return result

    def reverbAugment(self, sound):
        return [self.echo(sound, self.sr, 0.1, 1.5, 0.4, 0), self.echo(sound, self.sr, 0.1, 1.5, 0.4, 2)]

    def augmented_source(self):
        augments = [self.data]
        if self.label == 1:
            augments += self.add_noise()
            augments += self.pitch_shift(augments[0])
            augments += self.pitch_shift(augments[1])
            tmp = []
            for sound in augments:
                tmp += self.reverbAugment(sound)
            augments += tmp
        return augments
    
    def augmented(self):
        augments = [self.data]
        if self.label == 1:
            augments += self.add_noise()
            augments += self.pitch_shift(augments[0])
            augments += self.pitch_shift(augments[1])
            tmp = []
            for sound in augments:
                tmp += self.reverbAugment(sound)
            augments += tmp
        data = []
        for i in augments:
            data.append(self.get_energy(i))
        return data

    def get_energy(self, y=[]):
        if len(y) == 0:
            y = self.data
        y, _ = li.effects.trim(y)
        x = tf.keras.preprocessing.sequence.pad_sequences(
            [y],
            maxlen=int(self.sr * 1.0),
            padding='post',
            truncating='post',
            dtype='float32'
        )[0]
        coeff = sp.signal.firwin(999, [260, 700], fs=self.sr, pass_zero=False)
        x_filtered = sp.signal.lfilter(coeff, 1.0, x)
        x_normalized = x_filtered / x_filtered.max()
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


if __name__ == "__main__":
    pass
