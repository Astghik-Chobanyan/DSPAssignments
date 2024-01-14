import numpy as np
import matplotlib.pyplot as plt

from simple_wav import write_audio, read_audio


def gen_sinusoid(freq: int, duration: float, sample_rate: int, amplitude: int):
	N = duration * sample_rate
	t_samples = np.arange(N)
	omega = 2 * np.pi * freq / sample_rate # angular frequency
	sinusoid = amplitude * np.sin(omega * t_samples) # sinusoidal signal
	return sinusoid


def gen_triangular(freq: int, duration: float, sample_rate, amplitude: int):
	N = int(duration * sample_rate)
	t = np.linspace(0, duration, N, endpoint=False)
	triangular = amplitude * 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
	return triangular


def plot_ampl_spectr(data, sr, db_scale=False, epsilon=1e-3):
	# data: input fragment of real signal
	# sr: sampling rate of signal
	N = len(data)
	coef_no = int(N / 2) + 1 # coefficients number according to all frequencies
	delta_freq = sr / N # frequency resolution of according N len DFT basis frequencies
	# delta_freq = delta_omega / (2*pi) * sr = sr / N, where delta_omega = 2 * pi / N
	freqs = np.arange(coef_no) * delta_freq
	coefs = np.fft.rfft(data)
	# coefs = np.fft.fft2(data)
	amplitude_spectr = np.abs(coefs)
	amplitude_spectr_db = 10 * np.log10(amplitude_spectr + epsilon)
	if db_scale:
		plt.plot(freqs, amplitude_spectr_db)
	else:
		plt.plot(freqs, amplitude_spectr)
	plt.xlabel('freqs in hz')
	plt.ylabel('amplituds')
	plt.show()
	return None


if __name__ == '__main__':
	sample_rate_ = 16000  # hz
	freq_ = 1500  # hz
	amplitude_ = 1000
	duration_ = 0.06  # secs
	# generating sinusoid signal
	sinusoid = gen_sinusoid(freq_, duration_, sample_rate_, amplitude_)
	write_audio('data/sinusoid.wav', sample_rate_, sinusoid)
	plot_ampl_spectr(sinusoid, sample_rate_, db_scale=False)

	# generating triangular signal
	triangular = gen_triangular(freq_, duration_, sample_rate_, amplitude_)
	write_audio('data/triangular.wav', sample_rate_, triangular)
	plot_ampl_spectr(triangular, sample_rate_)

	sr, data = read_audio('data/v5.wav')
	plot_ampl_spectr(data[:10000], sr)
