"""
Build signals
"""
import wave

from pprint import pprint

import numpy as np
from numpy.lib import stride_tricks

import matplotlib.pyplot as plt

from scipy.io import wavfile


def waveform(audio_file: str, index: str = 'X'):
    """ draw signal wave file and sectrum of signal """
    _, signal_object = wavfile.read(audio_file)
    pprint('')
    pprint(signal_object)

    # Plot the signal read from wav file
    plt.subplot(211)

    # TODO truncate only file name
    plt.title(f'waveform of a wav file {audio_file}')
    plt.plot(signal_object)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.savefig(f'waveform{index}.png')


def spectogram(audio_file: str, index: str = 'X'):

    def graph_spectrogram(wav_file):
        sound_info, frame_rate = get_wav_info(wav_file)
        plt.figure(num=None, figsize=(19, 12))
        plt.subplot(111)
        plt.title('spectrogram of %r' % wav_file)
        plt.specgram(sound_info, Fs=frame_rate)
        plt.savefig(f'spectrogram{index}.png')

    def get_wav_info(wav_file):

        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        sound_info = np.fromstring(frames, 'int16')
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate

    return graph_spectrogram(audio_file)


def stft_spectogram(audio_file: str, index: str = 'X'):

    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        """ short time fourier transform of audio signal """
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
        # cols for windowing
        cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(
            samples.strides[0]*hopSize, samples.strides[0])).copy()
        frames *= win

        return np.fft.rfft(frames)

    def logscale_spec(spec, sr=44100, factor=20.):
        """ scale frequency axis logarithmically """
        timebins, freqbins = np.shape(spec)

        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins-1)/max(scale)
        scale = np.unique(np.round(scale))

        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
            else:
                newspec[:, i] = np.sum(
                    spec[:, int(scale[i]):int(scale[i+1])], axis=1)

        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

        return newspec, freqs

    def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="jet"):
        """ plot spectrogram"""
        samplerate, samples = wavfile.read(audiopath)

        s = stft(samples, binsize)

        sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

        ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel

        timebins, freqbins = np.shape(ims)

        print("timebins: ", timebins)
        print("freqbins: ", freqbins)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(ims), origin="lower", aspect="auto",
                   cmap=colormap, interpolation="none")
        plt.colorbar()

        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins])

        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in (
            (xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
        ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

        if plotpath:
            plt.savefig(plotpath, bbox_inches="tight")
        else:
            plt.savefig(f'spectrogram{index}.png')
    return plotstft(audio_file)
