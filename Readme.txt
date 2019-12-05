

Project Folder Details 
----------------------------------------------------------------------------------------------------------------------------------------

dataset_wav --- Corpus contains audio .wav files of spoken urdu words

dataset_java_feature --- The folder contains files saved with extracted MFCCs features using the librosa java implementation. MFCCs are saved in seperate text files with the same name as the audio file for each audio later to be used for comparison with the librosa python generated MFCCs.

librosa/librosa_java --- Folder has java implementation for librosa. Reads the audio .wav files to extract MFCCs and saves them in seperate text files for each audio.

librosa/librosa_python --- Folder has python librosa code to extract MFCCs from .wav audio

mfcc --- Code for extracting features using python_speech_features library (wav2mfcc.py) & using librosa library (wav2mfcc_librosa.py)

model --- Contains code for LSTM Model for classification of native and non-native speakers (native_classifier.py)

Others --- Contains potential code for feature extraction using java (eg. OpenIMAJ library) from other different libraries that were tried in the study.

utilities --- Creating Wav audio object in python ( data_loader.py )

train_model.py --- train model using MFCCs extracted using python librosa 

train_model_librosa.py --- train model using MFCCs extracted using librosa java implementation

---------------------------------------------------------------------------------------------------------------------------------------

Note: 

add instructions to run RMSE code


