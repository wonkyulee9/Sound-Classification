# This code produces a form of audio data called the "Autocorrelogram"
# Autocorrelogram is a 3d depth containing data, so it is very expensive to produce.
# This code sees the spectrogram of each audio and clusters each time frame to get the most distinct 5 time frames and creates a piece of the autocorrelogram for the corresponding 5 time frames.

import librosa.display
import numpy as np
import cupy as cp
import math
import csv
from sklearn.cluster import KMeans
from gammatone import filters


# cupy library is used to compute faster
cp.cuda.Device(3).use()

# create hann window with length 1024
win = cp.zeros([1024, ], dtype=float)


def hann(n, N):
    hann = (1 - math.cos(2 * math.pi * n / (N - 1))) / 2
    return hann


for x in range(1024):
    win[x] = hann(x, 1024)


# Open metadata
with open('UrbanSound8K/metadata/UrbanSound8K.csv') as mf:
    meta = list(csv.reader(mf))
mf.close()

acgdata = np.empty([0, 5, 128, 128])
atdata = np.empty([0, 5, 128, 128])
iter = 0

fold = 0

for x in range(1, len(meta)):

    if int(meta[x][5]) == fold + 1:
        filename, foldno, classID = meta[x][0], int(meta[x][5]), int(meta[x][6])

        s, sr = librosa.load('UrbanSound8K/audio/fold'+str(foldno)+'/'+filename, sr=44100)
        fcoefs = filters.make_erb_filters(sr, filters.centre_freqs(sr, 128, 40), odr=4)
        g = filters.erb_filterbank(s, fcoefs)
        c = cp.asarray(g)
        c = cp.power(c, 2)

        if len(c[0]) // 66536 > 0 and len(c[0]) % 65536 < 65536 / 2:
            nspecs = len(c[0]) // 65536
        else:
            nspecs = (len(c[0]) // 65536) + 1

        if len(c[0]) < 65536 + 512:
            c = cp.pad(c, ((0, 0), (0, 66048 - len(c[0]))), 'constant', constant_values=0)

        # For each spectrogram,
        for n in range(nspecs):
            acg = np.empty([5, 128, 128])

            # Clustering
            with open('/Spectrograms/specs.dat', 'rb') as sf:
                sp = np.load(sf)
            sf.close()
            sp = sp[foldno-1]
            
            a = np.rot90(sp[iter], 1, (1, 0)) + 1

            wasblank = False
            for t in range(123):
                if np.amax(a[127 - t]) == 0.0:
                    a = np.delete(a, 127 - t, 0)
                    wasblank = True
            if wasblank == True:
                a = np.delete(a, len(a) - 1, 0)

            ind = []
            if len(a) < 5:
                kmeans=KMeans(n_clusters=4, random_state=0).fit(a)
                for i in range(4):
                    d = kmeans.transform(a)[:, i]
                    ind.append(np.argsort(d)[::][0])
                ind.append(ind[3])
            else:
                kmeans = KMeans(n_clusters=5, random_state=0).fit(a)
                for i in range(5):
                    d = kmeans.transform(a)[:, i]
                    ind.append(np.argsort(d)[::][0])


            # Make 5 acg for each spectrogram
            for i, j in enumerate(ind):
                if n > 0 and n == nspecs-1:
                    for d in range(128):
                        seg1 = c[:, len(c[0])-(54436+512) + j * 512 + cp.arange(1024)] * win
                        seg2 = c[:, len(c[0])-(54436+512) + j * 512 + cp.arange(1024) + d * 8] * win
                        acg[i][d] = cp.asnumpy(cp.multiply(seg1, seg2).mean(1))
                else:
                    for d in range(128):
                        seg1 = c[:, n*65536 + j * 512 + cp.arange(1024)] * win
                        seg2 = c[:, n*65536 + j * 512 + cp.arange(1024) + d * 8] * win
                        acg[i][d] = cp.asnumpy(cp.multiply(seg1, seg2).mean(1))

                acg[i] = cp.transpose(acg[i])
                acg[i] = librosa.power_to_db(np.sqrt(acg[i]), ref=np.max)

            acg = acg/40 + 1
            acgdata = np.insert(acgdata, len(acgdata), acg, axis=0)
            if n == 0:
                atdata = np.insert(atdata, len(atdata), acg, axis=0)
            iter += 1

        print(x, ' audio file data processed')


with open('acg_train' + str(fold) + '.dat', 'wb') as af:
    np.save(af, acgdata)
af.close()
with open('acg_test' + str(fold) + '.dat', 'wb') as tef:
    np.save(tef, atdata)
tef.close()
