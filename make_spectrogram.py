import numpy as np
import librosa.display
import csv
import matplotlib.pyplot as plt

# This file will make a spectrogram with size 1024, hop length 512, sampling rate 44100, with 128 channels from 50 to 22500 Hz.
# Librosa library is used to make mel-spectrograms.
# For simplicity we assume each file to be of length 3s and pad as zeros if shorter.

with open('UrbanSound8K/metadata/UrbanSound8K.csv') as meta_csv:
    metadata = list(csv.reader(meta_csv))
meta_csv.close()

specs = [np.empty([0, 64, 173]) for _ in range(10)]

for x in range(1, len(metadata)):
    filename, foldno, classID = metadata[x][0], int(metadata[x][5]), metadata[x][6]

    sig, sr = librosa.load('UrbanSound8K/audio/fold'+str(foldno)+'/'+filename, sr=44100)

    spec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=1024, hop_length=1024, n_mels=64, fmax=22050)

    if spec.shape[1] < 173:
        spec = np.pad(spec, ((0, 0), (0, 173-len(S[0]))), 'constant', constant_values=0)

    db_spec = librosa.power_to_db(spec, ref=np.max)
    specs[foldno-1] = np.insert((specs[foldno-1] / 40.0) + 1, len(specs[foldno-1]), db_spec[:, 0:173], axis=0)
    print(x, "audio files processed")
    
    '''
    # This block shows the spectrogram visually to check that it has been properly made
    librosa.display.specshow(db_spec, sr=44100, y_axis='mel', fmax=22050, x_axis='time')
    print(S.shape)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()
    '''
    
for x in range(10):
    with open('/Spectrograms/spec'+str(x)+'.dat', 'wb') as spf:
        np.save(spf, specs[x])
    spf.close()
