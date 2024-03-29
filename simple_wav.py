import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import subprocess

file_path = 'v.wav'
start = 16000
end = 24000


def read_audio(path):
	# path: path of audio file with .wav extansion
	sr, data = wavfile.read(path)
	data = data.astype('float32')
	return sr, data


def write_audio(path, sr, data):
	# path: the path of output file,
	# sr: sampling rate
	# data: the audio waveform data
	data = data.astype('int16')
	wavfile.write(path, sr, data)
	return None


def compute_power(data):
	power = np.mean(data**2)
	return power


def plot_waveform(data, sampling_frequency=None):
	# data: the audio waveform data
	if sampling_frequency:
		sampling_period = 1 / sampling_frequency
		time_points = sampling_period * np.arange(len(data))
		plt.plot(time_points, data)
		plt.ylabel('amplitude')
		plt.xlabel('time [s]')
		plt.show()
	else:
		plt.plot(data)
		plt.ylabel('amplitude')
		plt.xlabel('samples')
		plt.show()
	return None


def change_sampling_rate(input_path, output_path, output_sr):
	# input_path: path of input audio file
	# output_path: path of output audio file
	# output_sr: sampling rate of autput_path, integer
	assert input_path != output_path, 'input_path and output_path should not coincide'
	# make sure you have installed ffmpeg
	subprocess.call(['ffmpeg', '-i', input_path, '-ar', str(output_sr), output_path])
	return None
