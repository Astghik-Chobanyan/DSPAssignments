import numpy as np
from simple_wav import write_audio


def karplus_strong(x, a, N):
    """
    Karplus-Strong algorithm to generate a plucked string sound.

    Parameters:
    x (np.array): Input vector x (initial noise burst)
    a (float): Decay factor, must be between 0 and 1
    N (int): Length of the output signal

    Returns:
    np.array: Generated output signal y[n]
    """
    M = len(x)
    y = np.zeros(N)
    y[:M] = x

    for n in range(M, N):
        y[n] = a**(n // M) * x[n % M]

    return y


def generate_plucked_string_sound(a, N, filename):
    """
    Generates a random x vector, feeds it to the Karplus-Strong algorithm,
    and saves the output signal as a .wav file.

    Parameters:
    a (float): Decay factor for the Karplus-Strong algorithm
    N (int): Length of the output signal
    filename (str): Filename to save the .wav file

    Returns:
    None
    """
    x = (2 * np.random.randint(0, 2, 200) - 1).astype(float)

    y = karplus_strong(x, a, N)

    y_normalized = np.int16(y / np.max(np.abs(y)) * 32767)
    write_audio(filename, N, y_normalized)


if __name__ == '__main__':
    print("Generating plucked string sound using Karplus Strong algorithm")
    # you can change a and N values and get different results
    generate_plucked_string_sound(a=0.7, N=20000, filename='data/plucked_string.wav')
