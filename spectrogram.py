import os
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io.wavfile as wavfile

from scipy import signal
from simple_wav import read_audio

DTMF_Tones = {
	'1': (697, 1209),
	'2': (697, 1336),
	'3': (697, 1477),
	'A': (697, 1633),
	'4': (770, 1209),
	'5': (770, 1336),
	'6': (770, 1477),
	'B': (770, 1633),
	'7': (852, 1209),
	'8': (852, 1336),
	'9': (852, 1477),
	'C': (852, 1633),
	'*': (941, 1209),
	'0': (941, 1336),
	'#': (941, 1477),
	'D': (941, 1633)
}


def get_spcgr(audio_path, cmapval, short_time, size_pixel_x, size_pixel_y, dpi_value,
			  start=0, duration=5, epsilon=1e-3):
	'''
	Here we use scipy.signal.spectrogram, can be useful to check the manual implementation of spectrogram, 
	it also saves the spectogram as a picture
	'''
	# extracts audio data, then saves spectrogram of duration sec
	fs, mix_data = wavfile.read(audio_path)
	audio_dir = os.path.dirname(audio_path)
	audio_name = os.path.basename(audio_path).split('.') [0]
	frlen = int(short_time * fs / 1000) # frame length with duration short_time ms 
	ovlap = int(frlen / 2) # overlap of adjacent frames
	f, t, Sxx = signal.spectrogram(mix_data[start*fs:(start+duration)*fs], fs=fs, window='hamming', nperseg=frlen, noverlap=ovlap)
	fig = plt.figure()
	# fig.canvas.set_window_title(audio_name)
	#plt.pcolormesh(t, f, Sxx, cmap=cmapval, norm=colors.PowerNorm(gamma=1./3.), edgecolors='None', shading='gouraud') # cmap = 'inferno', 'bwr', 'binary', 'Greys', 'OrRd', 'jet', 'Blues', 'afmhot', 'RdYlBu', 'PuBu', 'PuBu_r'
	plt.pcolormesh(t, f, Sxx, cmap=cmapval, norm=colors.LogNorm(vmin=epsilon, vmax=Sxx.max()), edgecolors='None', shading='gouraud')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	fig.set_size_inches(size_pixel_y / dpi_value, size_pixel_x / dpi_value)
	fig.savefig(os.path.join(audio_dir, audio_name + '.png'), dpi = dpi_value)
	plt.close()
	# plt.show()
	return None
	

def plot_spectrograms_from_wavs(wavs_dir):
	file_paths = glob.glob(os.path.join(wavs_dir, "*.wav"))
	short_time_ = 32  # miliseconds
	start_t = 0 # start time in seconds
	dur = 7 # duration in seconds
	color_theme = 'hot' # inferno, gnuplot2, jet, jet_r, Greys_r, terrain, plasma, gist_gray, gist_gray_r, spectral, bone_r, hot
	dpi_value = 96 # dots per inch
	size_pixel_y = 1301
	size_pixel_x = 710
	for path in file_paths:
		get_spcgr(audio_path=path, cmapval=color_theme, start=start_t, duration=dur,
				  short_time=short_time_, size_pixel_x=size_pixel_x,
				  size_pixel_y=size_pixel_y, dpi_value=dpi_value)
		print('spectrogram of \"{}\"saved'.format(os.path.basename(path)))
		plt.show()


def create_sprectrogram(input_signal, frame_len, step_size, window_f):
	num_frames = 1 + int((len(input_signal) - frame_len) / step_size)
	window = window_f(frame_len)
	spectrogram = np.zeros((int(frame_len / 2 + 1), num_frames))

	for i in range(num_frames):
		start_idx = i * step_size
		end_idx = start_idx + frame_len
		frame = input_signal[int(start_idx):int(end_idx)] * window
		frame_fft = np.fft.rfft(frame)
		spectrogram[:, i] = np.abs(frame_fft)

	return spectrogram


def plot_spectrogram(spectrogram, sampling_rate, input_signal_len, color_theme='magma'):
	plt.figure(figsize=(10, 6))
	plt.imshow(spectrogram, aspect='auto', origin='lower',
			   extent=[0, input_signal_len / sampling_rate, 0, sampling_rate / 2], cmap=color_theme)
	plt.colorbar(label='Magnitude')
	plt.xlabel('Time [s]')
	plt.ylabel('Frequency [Hz]')
	plt.title('Spectrogram')
	plt.show()


def generate_tone(digit, duration, sampling_rate):
	if digit not in DTMF_Tones:
		raise ValueError(f"Digit {digit} is not valid for DTMF tones.")
	low_freq, high_freq = DTMF_Tones[digit]
	t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
	tone = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * high_freq * t)
	return (tone / 2)  # Normalizing the amplitude


def generate_phone_number(phone_number, save_path):
	dtmf_signal = np.array([], dtype=np.float32)
	sampling_rate = 8000
	duration = 0.6
	pause = np.zeros(int(sampling_rate * duration))
	for digit in phone_number:
		tone = generate_tone(digit, duration, sampling_rate)
		dtmf_signal = np.concatenate((dtmf_signal, tone, pause))
	scipy.io.wavfile.write(save_path, sampling_rate, dtmf_signal)


if __name__ == '__main__':
	short_time = 32
	# create and save spectrograms of all wav files in given directory
	plot_spectrograms_from_wavs(wavs_dir='data/')

	# create a spectrogram of given input audio and plot it
	sr, input_signal = read_audio('data/v5.wav')
	spec = create_sprectrogram(input_signal=input_signal,
							   frame_len=int(short_time * sr / 1000),
							   step_size=10, window_f=np.hanning)
	plot_spectrogram(spectrogram=spec, input_signal_len=len(input_signal), sampling_rate=sr)

	# generate DTMF signal of given phone number
	phone_number = '077322387'
	generate_phone_number(phone_number=phone_number, save_path='data/phone_number.wav')

	# plot the spectrogram of phone number
	phone_sr, phone_audio = read_audio('data/phone_number.wav')
	phone_spect = create_sprectrogram(input_signal=phone_audio,
									  frame_len=int(short_time * phone_sr) / 1000,
									  step_size=10, window_f=np.hanning)
	plot_spectrogram(spectrogram=phone_spect, input_signal_len=len(phone_audio), sampling_rate=phone_sr)
