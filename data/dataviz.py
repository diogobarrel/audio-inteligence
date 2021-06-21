""" data vizualisation """
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

__HOP_SIZE = 256

def plot(audio_file, index='plot'):
    fig, plots = plt.subplots(nrows=3, ncols=1, sharex=True)

    wave_plot = plots[0]
    wave_plot.set(title=f'{audio_file.name} waveform')
    wave_plot.set(xlabel='Sample')
    wave_plot.set(ylabel='Amplitude')
    librosa.display.waveplot(audio_file.data, audio_file.sample_rate, ax=wave_plot)
    wave_plot.label_outer()

    y_scale = np.abs(audio_file.stft) ** 2
    D = librosa.amplitude_to_db(y_scale, ref=np.max)

    linear_spec_plot = plots[1]
    linear_spec_plot.set(
        title=f'{audio_file.name} Linear-frequency power spectrogram')
    img = librosa.display.specshow(
        D, sr=audio_file.sample_rate,
        hop_length=__HOP_SIZE,
        x_axis='time',
        y_axis='linear',
        ax=linear_spec_plot)
    linear_spec_plot.label_outer()

    log_spec_plot = plots[2]
    log_spec_plot.set(title=f'{audio_file.name} Log-frequency power spectrogram')
    librosa.display.specshow(D, y_axis='log',
                                sr=audio_file.sample_rate,
                                hop_length=__HOP_SIZE,
                                x_axis='time',
                                ax=log_spec_plot)
    log_spec_plot.label_outer()

    fig.tight_layout()
    fig.colorbar(img, ax=plots, format="%+2.f dB")
    filename = audio_file.name.replace('.wav', '')
    fig.savefig(f'{index}_{filename}.png')
