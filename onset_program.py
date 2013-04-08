#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012, 2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

"""
Please note that this program released together with the paper

"Evaluating the Online Capabilities of Onset Detection Methods"
by Sebastian Böck, Florian Krebs and Markus Schedl
in Proceedings of the 13th International Society for Music Information
Retrieval Conference (ISMIR), 2012

is not tuned in any way for speed/memory efficiency. However, it can be used
to compare other onset detection algorithms with the method described in our
paper. All results given in the paper were obtained with this code and
evaluated with the provided onset_evaluation.py script.

If you use this software, please cite the above paper.

Please send any comments, enhancements, errata, etc. to the main author.

"""

import numpy as np
import scipy.fftpack as fft
from scipy.io import wavfile


class Filter(object):
    """
    Filter Class.

    """
    def __init__(self, ffts, fs, bands=12, fmin=27.5, fmax=16000, equal=False):
        """
        Creates a new Filter object instance.

        :param ffts: number of FFT coefficients
        :param fs: sample rate of the audio file
        :param bands: number of filter bands [default=12]
        :param fmin: the minimum frequency [in Hz, default=27.5]
        :param fmax: the maximum frequency [in Hz, default=16000]
        :param equal: normalize each band to equal energy [default=False]

        """
        # samplerate
        self.fs = fs
        # reduce fmax if necessary
        if fmax > fs / 2:
            fmax = fs / 2
        # get a list of frequencies
        frequencies = self.frequencies(bands, fmin, fmax)
        # conversion factor for mapping of frequencies to spectrogram bins
        factor = (fs / 2.0) / ffts
        # map the frequencies to the spectrogram bins
        frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
        # only keep unique bins
        frequencies = np.unique(frequencies)
        # filter out all frequencies outside the valid range
        frequencies = [f for f in frequencies if f < ffts]
        # number of bands
        bands = len(frequencies) - 2
        assert bands >= 3, "cannot create filterbank with less than 3 frequencies"
        # init the filter matrix with size: ffts x filter bands
        self.filterbank = np.zeros([ffts, bands], dtype=np.float)
        # process all bands
        for band in range(bands):
            # edge & center frequencies
            start, mid, stop = frequencies[band:band + 3]
            # create a triangular filter
            self.filterbank[start:stop, band] = self.triang(start, mid, stop, equal)

    @staticmethod
    def frequencies(bands, fmin, fmax, a=440):
        """
        Returns a list of frequencies aligned on a logarithmic scale.

        :param bands: number of filter bands per octave
        :param fmin: the minimum frequency [in Hz]
        :param fmax: the maximum frequency [in Hz]
        :param a: frequency of A0 [in Hz, default=440]
        :return a list of frequencies
        Using 12 bands per octave and a=440 corresponding to the MIDI notes.

        """
        # factor 2 frequencies are apart
        factor = 2.0 ** (1.0 / bands)
        # start with A0
        freq = a
        frequencies = [freq]
        # go upwards till fmax
        while freq <= fmax:
            # multiply once more, since the included frequency is a frequency
            # which is only used as the right corner of a (triangular) filter
            freq *= factor
            frequencies.append(freq)
        # restart with a and go downwards till fmin
        freq = a
        while freq >= fmin:
            # divide once more, since the included frequency is a frequency
            # which is only used as the left corner of a (triangular) filter
            freq /= factor
            frequencies.append(freq)
        # sort frequencies
        frequencies.sort()
        # return the list
        return frequencies

    @staticmethod
    def triang(start, mid, stop, equal=False):
        """
        Calculates a triangular window of the given size.

        :param start: starting bin (with value 0, included in the returned filter)
        :param mid: center bin (of height 1, unless norm is True)
        :param stop: end bin (with value 0, not included in the returned filter)
        :param equal: normalize the area of the filter to 1 [default=False]
        :return a triangular shaped filter

        """
        # height of the filter
        height = 1.
        # normalize the height
        if equal:
            height = 2. / (stop - start)
        # init the filter
        triang_filter = np.empty(stop - start)
        # rising edge
        triang_filter[:mid - start] = np.linspace(0, height, (mid - start), endpoint=False)
        # falling edge
        triang_filter[mid - start:] = np.linspace(height, 0, (stop - mid), endpoint=False)
        # return
        return triang_filter


class Wav(object):
    """
    Wav Class is a simple wrapper around scipy.io.wavfile.

    """
    def __init__(self, filename):
        """
        Creates a new Wav object instance of the given file.

        :param filename: name of the .wav file

        """
        # read in the audio
        self.samplerate, self.audio = wavfile.read(filename)
        # scale the audio values to the range -1...1 depending on the audio type
        self.audio = self.audio / float(np.iinfo(self.audio.dtype).max)
        # set the length
        self.samples = np.shape(self.audio)[0]
        # set the number of channels
        try:
            # multi channel files
            self.channels = np.shape(self.audio)[1]
        except IndexError:
            # catch mono files
            self.channels = 1

    def attenuate(self, attenuation):
        """
        Attenuate the audio signal.

        :param attenuation: attenuation level given in dB

        """
        self.audio /= np.power(np.sqrt(10.), attenuation / 10.)

    def downmix(self):
        """
        Down-mix the audio signal to mono.

        """
        if self.channels > 1:
            self.audio = np.sum(self.audio, -1) / self.channels

    def normalize(self):
        """
        Normalize the audio signal.

        """
        self.audio /= np.max(self.audio)


class Spectrogram(object):
    """
    Spectrogram Class.

    """
    def __init__(self, wav, window_size=2048, fps=200, online=True, phase=True):
        """
        Creates a new Spectrogram object instance and performs a STFT on the given audio.

        :param wav: a Wav object
        :param window_size: is the size for the window in samples [default=2048]
        :param fps: is the desired frame rate [default=200]
        :param online: work in online mode (i.e. use only past audio information) [default=True]
        :param phase: include phase information [default=True]

        """
        # init some variables
        self.wav = wav
        self.fps = fps
        # derive some variables
        self.hop_size = float(self.wav.samplerate) / float(self.fps)  # use floats so that seeking works properly
        self.frames = int(self.wav.samples / self.hop_size)
        self.ffts = int(window_size / 2)
        self.bins = int(window_size / 2)  # initial number equal to ffts, can change if filters are used
        # init STFT matrix
        self.stft = np.empty([self.frames, self.ffts], np.complex)
        # create windowing function
        self.window = np.hanning(window_size)
        # step through all frames
        for frame in range(self.frames):
            # seek to the right position in the audio signal
            if online:
                # step back a complete window_size after moving forward 1 hop_size
                # so that the current position is at the stop of the window
                seek = int((frame + 1) * self.hop_size - window_size)
            else:
                # step back half of the window_size so that the frame represents the centre of the window
                seek = int(frame * self.hop_size - window_size / 2)
            # read in the right portion of the audio
            if seek >= self.wav.samples:
                # stop of file reached
                break
            elif seek + window_size >= self.wav.samples:
                # stop behind the actual audio stop, append zeros accordingly
                zeros = np.zeros(seek + window_size - self.wav.samples)
                signal = self.wav.audio[seek:]
                signal = np.append(signal, zeros)
            elif seek < 0:
                # start before the actual audio start, pad with zeros accordingly
                zeros = np.zeros(-seek)
                signal = self.wav.audio[0:seek + window_size]
                signal = np.append(zeros, signal)
            else:
                # normal read operation
                signal = self.wav.audio[seek:seek + window_size]
            # multiply the signal with the window function
            signal = signal * self.window
            # only shift and perform complex DFT if needed
            if phase:
                # circular shift the signal (needed for correct phase)
                signal = fft.fftshift(signal)
            # perform DFT
            self.stft[frame] = fft.fft(signal, window_size)[:self.ffts]
            # next frame
        # magnitude spectrogram
        self.spec = np.abs(self.stft)
        # phase
        if phase:
            self.phase = np.arctan2(np.imag(self.stft), np.real(self.stft))

    # pre-processing stuff
    def aw(self, floor=5, relaxation=10):
        """
        Perform adaptive whitening on the magnitude spectrogram.

        :param floor: floor value [default=5]
        :param relaxation: relaxation time in seconds [default=10]

        "Adaptive Whitening For Improved Real-time Audio Onset Detection"
        Dan Stowell and Mark Plumbley
        Proceedings of the International Computer Music Conference (ICMC), 2007

        """
        mem_coeff = 10.0 ** (-6. * relaxation / self.fps)
        P = np.zeros_like(self.spec)
        # iterate over all frames
        for f in range(self.frames):
            spec_floor = np.maximum(self.spec[f], floor)
            if f > 0:
                P[f] = np.maximum(spec_floor, mem_coeff * P[f - 1])
            else:
                P[f] = spec_floor
        # adjust spec
        self.spec /= P

    def filter(self, filterbank=None):
        """
        Filter the magnitude spectrogram with a filterbank.

        :param filterbank: Filter object which includes the filterbank [default=None]

        If no filter is given a standard one will be created.

        """
        if filterbank is None:
            # construct a standard filterbank
            filterbank = Filter(ffts=self.ffts, fs=self.wav.samplerate).filterbank
        # filter the magnitude spectrogram with the filterbank
        self.spec = np.dot(self.spec, filterbank)
        # adjust the number of bins
        self.bins = np.shape(filterbank)[1]

    def log(self, mul=20, add=1):
        """
        Take the logarithm of the magnitude spectrogram.

        :param mul: multiply the magnitude spectrogram with given value [default=20]
        :param add: add the given value to the magnitude spectrogram [default=1]

        """
        if add <= 0:
            raise ValueError("a positive value must be added before taking the logarithm")
        self.spec = np.log10(mul * self.spec + add)


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection function
    based on the magnitude or phase information of a spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.22, frames=None):
        """
        Creates a new ODF object instance.

        :param spectrogram: the spectrogram on which the detections functions operate
        :param ratio: calculate the difference to the frame which has the given magnitude ratio [default=0.22]
        :param frames: calculate the difference to the N-th previous frame [default=None]

        """
        self.s = spectrogram
        # determine the number off diff frames
        if frames is None:
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.s.window > ratio)
            diff_samples = self.s.window.size / 2 - sample
            # convert to frames
            frames = int(round(diff_samples / self.s.hop_size))
        # set the minimum to 1
        if frames < 1:
            frames = 1
        self.diff_frames = frames

    @staticmethod
    def wraptopi(angle):
        """
        Wrap the phase information to the range -π...π.

        """
        return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi

    def diff(self, spec, pos=False, diff_frames=None):
        """
        Calculates the difference on the magnitude spectrogram.

        :param spec: the magnitude spectrogram
        :param pos: only keep positive values [default=False]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]

        """
        diff = np.zeros_like(spec)
        if diff_frames is None:
            diff_frames = self.diff_frames
        # calculate the diff
        diff[diff_frames:] = spec[diff_frames:] - spec[0:-diff_frames]
        if pos:
            diff = diff * (diff > 0)
        return diff

    # Onset Detection Functions
    def hfc(self):
        """
        High Frequency Content.

        "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
        Paul Masri
        PhD thesis, University of Bristol, 1996

        """
        # HFC weights the magnitude spectrogram by the bin number, thus emphasising high frequencies
        return np.mean(self.s.spec * np.arange(self.s.bins), axis=1)

    def sd(self):
        """
        Spectral Diff.

        "A hybrid approach to musical note onset detection"
        Chris Duxbury, Mark Sandler and Matthew Davis
        Proceedings of the 5th International Conference on Digital Audio Effects (DAFx-02), 2002.

        """
        # Spectral diff is the sum of all squared positive 1st order differences
        return np.sum(self.diff(self.s.spec, pos=True) ** 2, axis=1)

    def sf(self):
        """
        Spectral Flux.

        "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
        Paul Masri
        PhD thesis, University of Bristol, 1996

        """
        # Spectral flux is the sum of all positive 1st order differences
        return np.sum(self.diff(self.s.spec, pos=True), axis=1)

    def mkl(self, epsilon=0.000001):
        """
        Modified Kullback-Leibler.

        :param epsilon: add epsilon to avoid division by 0 [default=0.000001]

        we use the implenmentation presented in:
        "Automatic Annotation of Musical Audio for Interactive Applications"
        Paul Brossier
        PhD thesis, Queen Mary University of London, 2006

        instead of the original work:
        "Onset Detection in Musical Audio Signals"
        Stephen Hainsworth and Malcolm Macleod
        Proceedings of the International Computer Music Conference (ICMC), 2003

        """
        if epsilon <= 0:
            raise ValueError("a positive value must be added before division")
        mkl = np.zeros_like(self.s.spec)
        mkl[1:] = self.s.spec[1:] / (self.s.spec[0:-1] + epsilon)
        # note: the original MKL uses sum instead of mean, but the range of mean is much more suitable
        return np.mean(np.log(1 + mkl), axis=1)

    def _pd(self):
        """
        Helper method used by pd() & wpd().

        """
        pd = np.zeros_like(self.s.phase)
        # instantaneous frequency is given by the first difference ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
        # change in instantaneous frequency is given by the second order difference ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
        pd[2:] = self.s.phase[2:] - 2 * self.s.phase[1:-1] + self.s.phase[:-2]
        # map to the range -pi..pi
        return self.wraptopi(pd)

    def pd(self):
        """
        Phase Deviation.

        "On the use of phase and energy for musical onset detection in the complex domain"
        Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
        IEEE Signal Processing Letters, Volume 11, Number 6, 2004

        """
        # take the mean of the absolute changes in instantaneous frequency
        return np.mean(np.abs(self._pd()), axis=1)

    def wpd(self):
        """
        Weighted Phase Deviation.

        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        # make sure the spectrogram is not filtered before
        assert np.shape(self.s.phase) == np.shape(self.s.spec)
        # wpd = spec * pd
        return np.mean(np.abs(self._pd() * self.s.spec), axis=1)

    def nwpd(self, epsilon=0.000001):
        """
        Normalized Weighted Phase Deviation.

        :param epsilon: add epsilon to avoid division by 0 [default=0.000001]

        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        if epsilon <= 0:
            raise ValueError("a positive value must be added before division")
        # normalize WPD by the sum of the spectrogram (add a small amount so that we don't divide by 0)
        return self.wpd() / np.add(np.mean(self.s.spec, axis=1), epsilon)

    def _cd(self):
        """
        Helper method used by cd() & rcd().

        we use the simple implementation presented in:
        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        assert np.shape(self.s.phase) == np.shape(self.s.spec)  # make sure the spectrogram is not filtered before
        # expected spectrogram
        cd_target = np.zeros_like(self.s.phase)
        # assume constant phase change
        cd_target[1:] = 2 * self.s.phase[1:] - self.s.phase[:-1]
        # add magnitude
        cd_target = self.s.spec * np.exp(1j * cd_target)
        # complex spectrogram
        # note: construct new instead of using self.stft, because pre-processing could have been applied
        cd = self.s.spec * np.exp(1j * self.s.phase)
        # subtract the target values
        cd[1:] -= cd_target[:-1]
        return cd

    def cd(self):
        """
        Complex Domain.

        "On the use of phase and energy for musical onset detection in the complex domain"
        Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
        IEEE Signal Processing Letters, Volume 11, Number 6, 2004

        """
        # take the sum of the absolute changes
        return np.sum(np.abs(self._cd()), axis=1)

    def rcd(self):
        """
        Rectified Complex Domain.

        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        # rectified complex domain
        rcd = self._cd()
        # only keep values where the magnitude rises
        rcd[1:] = rcd[1:] * (self.s.spec[1:] > self.s.spec[:-1])
        # take the sum of the absolute changes
        return np.sum(np.abs(rcd), axis=1)


class Onsets(object):
    """
    Onset Class.

    """
    def __init__(self, activations, fps, online=True):
        """
        Creates a new Onset object instance with the given activations of the
        ODF (OnsetDetectionFunction). The activations can be read in from a file.

        :param activations: an array containing the activations of the ODF
        :param fps: frame rate of the activations
        :param online: work in online mode (i.e. use only past information) [default=True]

        """
        self.activations = None     # activations of the ODF
        self.fps = fps              # framerate of the activation function
        self.online = online        # online peak-picking
        self.detections = []        # list of detected onsets (in seconds)
        # set / load activations
        if isinstance(activations, np.ndarray):
            # activations are given as an array
            self.activations = activations
        else:
            # read in the activations from a file
            self.load(activations)

    def detect(self, threshold, combine=30, pre_avg=100, pre_max=30, post_avg=30, post_max=70, delay=0):
        """
        Detects the onsets.

        :param threshold: threshold for peak-picking
        :param combine: only report 1 onset for N miliseconds [default=30]
        :param pre_avg: use N miliseconds past information for moving average [default=100]
        :param pre_max: use N miliseconds past information for moving maximum [default=30]
        :param post_avg: use N miliseconds future information for moving average [default=0]
        :param post_max: use N miliseconds future information for moving maximum [default=40]
        :param delay: report the onset N miliseconds delayed [default=0]

        In online mode, post_avg and post_max are set to 0.

        Implements the peak-picking method described in:

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2012

        """
        import scipy.ndimage as sim
        # online mode?
        if self.online:
            post_max = 0
            post_avg = 0
        # convert timing information to frames
        pre_avg = int(round(self.fps * pre_avg / 1000.))
        pre_max = int(round(self.fps * pre_max / 1000.))
        post_max = int(round(self.fps * post_max / 1000.))
        post_avg = int(round(self.fps * post_avg / 1000.))
        # convert to seconds
        combine /= 1000.
        delay /= 1000.
        # init detections
        self.detections = []
        # moving maximum
        max_length = pre_max + post_max + 1
        max_origin = int(np.floor((pre_max - post_max) / 2))
        mov_max = sim.filters.maximum_filter1d(self.activations, max_length, mode='constant', origin=max_origin)
        # moving average
        avg_length = pre_avg + post_avg + 1
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        mov_avg = sim.filters.uniform_filter1d(self.activations, avg_length, mode='constant', origin=avg_origin)
        # detections are activation equal to the maximum
        detections = self.activations * (self.activations == mov_max)
        # detections must be greater or equal than the moving average + threshold
        detections = detections * (detections >= mov_avg + threshold)
        # convert detected onsets to a list of timestamps
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            onset = float(i) / float(self.fps) + delay
            # only report an onset if the last N miliseconds none was reported
            if onset > last_onset + combine:
                self.detections.append(onset)
                # save last reported onset
                last_onset = onset

    def write(self, filename):
        """
        Write the detected onsets to the given file.

        :param filename: the target file name

        Only useful if detect() was invoked before.

        """
        with open(filename, 'w') as f:
            for pos in self.detections:
                f.write(str(pos) + '\n')

    def save(self, filename):
        """
        Save the onset activations to the given file.

        :param filename: the target file name

        """
        self.activations.tofile(filename)

    def load(self, filename):
        """
        Load the onset activations from the given file.

        :param filename: the target file name

        """
        self.activations = np.fromfile(filename)


def parser():
    import argparse
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software detects all onsets in
    the given files in online mode according to the method proposed in:

    "Evaluating the Online Capabilities of Onset Detection Methods"
    by Sebastian Böck, Florian Krebs and Markus Schedl
    in Proceedings of the 13th International Society for
    Music Information Retrieval Conference (ISMIR), 2012

    """)
    # general options
    p.add_argument('files', metavar='files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='store_true', help='be verbose')
    p.add_argument('-s', dest='save', action='store_true', default=False, help='save the activations of the onset detection functions')
    p.add_argument('-l', dest='load', action='store_true', default=False, help='load the activations of the onset detection functions')
    # online / offline mode
    p.add_argument('--offline', dest='online', action='store_false', default=True, help='operate in offline mode')
    # wav options
    wav_opts = p.add_argument_group('audio arguments')
    wav_opts.add_argument('--norm', action='store_true', default=None, help='normalize the audio [switches to offline mode]')
    wav_opts.add_argument('--att', action='store', type=float, default=None, help='attenuate the audio by ATT dB')
    # spectrogram options
    spec_opts = p.add_argument_group('spectrogram arguments')
    spec_opts.add_argument('--fps', action='store', default=200, type=int, help='frames per second')
    spec_opts.add_argument('--window', action='store', type=int, default=2048, help='DFT window length')
    spec_opts.add_argument('--ratio', action='store', type=float, default=0.22, help='window magnitude ratio to calc number of diff frames')
    spec_opts.add_argument('--frames', action='store', type=int, default=None, help='diff frames')
    # pre-processing
    pre_opts = p.add_argument_group('pre-processing arguments')
    # aw
    pre_opts.add_argument('--aw', action='store_true', default=False, help='apply adaptive whitening')
    pre_opts.add_argument('--floor', action='store', type=float, default=5.0, help='floor value for adaptive whitening [default=5.0]')
    pre_opts.add_argument('--relax', action='store', type=float, default=10.0, help='relaxation time for adaptive whitening [default=10.0]')
    # filter
    pre_opts.add_argument('--filter', action='store_true', default=None, help='filter the magnitude spectrogram with a filterbank')
    pre_opts.add_argument('--fmin', action='store', default=27.5, type=float, help='minimum frequency of filter in Hz [default=27.5]')
    pre_opts.add_argument('--fmax', action='store', default=16000, type=float, help='maximum frequency of filter in Hz [default=16000]')
    pre_opts.add_argument('--bands', action='store', type=int, default=12, help='number of bands per octave')
    pre_opts.add_argument('--equal', action='store_true', default=False, help='equalize triangular windows to have equal area')
    # logarithm
    pre_opts.add_argument('--log', action='store_true', default=None, help='logarithmic magnitude')
    pre_opts.add_argument('--mul', action='store', default=1, type=float, help='multiplier (before taking the log) [default=1]')
    pre_opts.add_argument('--add', action='store', default=1, type=float, help='value added (before taking the log) [default=1]')
    # onset detection
    onset_opts = p.add_argument_group('onset detection arguments')
    onset_opts.add_argument('-o', dest='odf', action='append', default=[], help='use this onset detection function (can be used multiple times) [hfc,sd,sf,mkl,pd,wpd,nwpd,cd,rcd,all]')
    onset_opts.add_argument('-t', dest='threshold', action='store', type=float, default=2.5, help='detection threshold')
    onset_opts.add_argument('--combine', action='store', type=float, default=30, help='combine onsets within N miliseconds [default=30]')
    onset_opts.add_argument('--pre_avg', action='store', type=float, default=100, help='build average over N previous miliseconds [default=100]')
    onset_opts.add_argument('--pre_max', action='store', type=float, default=30, help='search maximum over N previous miliseconds [default=30]')
    onset_opts.add_argument('--post_avg', action='store', type=float, default=70, help='build average over N following miliseconds [default=70]')
    onset_opts.add_argument('--post_max', action='store', type=float, default=30, help='search maximum over N following miliseconds [default=30]')
    onset_opts.add_argument('--delay', action='store', type=float, default=0, help='report the onsets N miliseconds delayed [default=0]')
    # version
    p.add_argument('--version', action='version', version='%(prog)s 1.04 (2013-02-27)')
    # parse arguments
    args = p.parse_args()

    # list of offered ODFs
    methods = ['hfc', 'sd', 'sf', 'mkl', 'pd', 'wpd', 'nwpd', 'cd', 'rcd']
    # use default values if no ODF is given
    if args.odf == []:
        args.odf = ['sf']
        if args.log is None:
            args.log = True
        if args.filter is None:
            args.filter = True
    # use all onset detection functions
    if 'all' in args.odf:
        args.odf = methods
    # remove not implemented/mistyped methods
    args.odf = list(set(args.odf) & set(methods))
    assert args.odf, 'at least one onset detection function must be given'
    # check if we need the STFT phase information
    if set(args.odf) & set(['pd', 'wpd', 'nwpd', 'cd', 'rcd']):
        args.phase = True
    else:
        args.phase = False

    # print arguments
    if args.verbose:
        print args

    # return args
    return args


def main():
    import os.path
    import glob
    import fnmatch

    # parse arguments
    args = parser()

    # determine the files to process
    files = []
    for f in args.files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*.wav')
        else:
            # file was given, append to list
            files.append(f)

    # only process .wav files
    files = fnmatch.filter(files, '*.wav')
    files.sort()

    # init filterbank
    filt = None

    # process the files
    for f in files:
        if args.verbose:
            print f

        # use the name of the file without the extension
        filename = os.path.splitext(f)[0]

        # do the processing stuff unless the activations are loaded from file
        if not args.load:
            # open the wav file
            w = Wav(f)
            # normalize audio
            if args.norm:
                w.normalize()
                args.online = False  # switch to offline mode
            # downmix to mono
            if w.channels > 1:
                w.downmix()
            # attenuate signal
            if args.att:
                w.attenuate(args.att)

            # spectrogram
            s = Spectrogram(w, args.window, args.fps, args.online, args.phase)
            # adaptive whitening
            if args.aw:
                s.aw(args.floor, args.relax)
            # filter
            if args.filter:
                # (re-)create filterbank if the samplerate of the audio changes
                if (filt is None) or (filt.fs != w.samplerate):
                    filt = Filter(args.window / 2, w.samplerate, args.bands, args.fmin, args.fmax, args.equal)
                # filter the spectrogram
                s.filter(filt.filterbank)
            # log
            if args.log:
                s.log(args.mul, args.add)

        # process all onset detection functions
        for odf in args.odf:
            # load the activations from file
            if args.load:
                o = Onsets("%s.onsets.%s" % (filename, odf), args.fps, args.online)
                pass
            else:
                # use the spectrogram to create an SpectralODF object
                sodf = SpectralODF(s, args.ratio, args.frames)
                # perform detection function on the object
                act = getattr(sodf, odf)()
                # create an Onset object with the returned activations
                o = Onsets(act, args.fps, args.online)
                if args.save:
                    # save the raw ODF activations
                    o.save("%s.onsets.%s" % (filename, odf))
                    # do not proceed with onset detection
                    continue
            # detect the onsets
            o.detect(args.threshold, args.combine, args.pre_avg, args.pre_max, args.post_avg, args.post_max, args.delay)
            # write the onsets to a file
            if len(args.odf) > 1:
                # include the ODF name
                o.write("%s.onsets.%s.txt" % (filename, odf))
            else:
                o.write("%s.onsets.txt" % (filename))
            if args.verbose:
                print 'detections:', o.detections
            # continue with next onset detection function
        # continue with next file

if __name__ == '__main__':
    main()
