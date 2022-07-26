import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pyaudio
import wave
from audio import get_audio
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


def predict(path1):
    filename = "weights.pkl"
    loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

    feature = extract_feature(path1, mfcc=True, chroma=True, mel=True)

    feature=feature.reshape(1,-1)

    prediction=loaded_model.predict(feature)
    print(prediction)


choice = input("Select 1 to record your voice or 2 to input a filename: ")

if choice == '1':
    print('Recording has started........')
    
    get_audio()
    
    path = 'Datasets/recorded/output.wav'
    predict(path)

elif choice== '2':
    path = input('write file name: ')
    predict(path)

else:
    print('Try again')