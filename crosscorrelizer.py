#!/usr/bin/env python
# coding: utf-8

""" 
    This script implements cross-correlation-based sound-source localization. 

    Use the class CrossCorrelizer to compute the cross-correlation between two
    channels in a wav file.

    Use Localizer to localize a signal given data about typical 
    cross-correlationgs at different angles (see the accompanying learn.py).

"""

from scipy.io import wavfile
import numpy as np

class CrossCorrelizer(object):
    """ 
        Implements computing cross-correlation of two signals at different time
        shifts.
    """

    def __init__(self, sample_rate, shift_max, shift_steps):
        """
            sample_rate : (int) the sample rate to expect.
            shift_max   : (int) by how much to shift signals wrt. each other 
                          in either direction (in samples).
            shift_steps : (int) shift the signal in steps of size <shift_steps>.
        """
        self.sample_rate = sample_rate
        self.shift_max   = shift_max
        self.shift_steps = shift_steps

    def __ccr__(self, left, right):
        """
            Normalized cross-correlation (in contrast to np.correlate()).
        """
        return ((left - left.mean()) * (right - right.mean())).mean() / left.std() / right.std()
    
    def cross_correlize(self, infile):
        """
            Compute the cross-correlation between the channels in <infile> for 
            different time shifts.

            infile may be a file name or a (N x 2) array.
        """
        if isinstance(infile, str):
            sr,infile = wavfile.read(infile)
            assert sr == self.sample_rate, \
                "Sample rate needs to be %s (is %s)" % (self.sample_rate, sr)

        left = infile[:,0]
        right = infile[:,1]

        shiftright  = [self.__ccr__(left[:-shift], right[shift:]) 
                        for shift in 
                            range(self.shift_steps,self.shift_max,self.shift_steps)]

        no_shift   = [self.__ccr__(left, right)]

        shiftleft = [self.__ccr__(right[:-shift], left[shift:]) 
                        for shift in 
                            range(self.shift_steps,self.shift_max,self.shift_steps)]

        result = shiftleft[::-1] + no_shift + shiftright 
        assert len(result) % 2 == 1

        return np.array(result)


def cross_correlizer(sample_rate, max_itd, max_frequency):
    """
        Convenience function for creating a CrossCorrelizer with appropriate
        parameters.

        sample_rate   : the sample rate of the wav files to expect.
        max_itd       : the maximum interaural time difference to test.
        max_frequency : the highest frequency to test.
    """
    shift_max = int(np.ceil(max_itd * sample_rate))
    shift_steps = int(float(sample_rate) / max_frequency / 2.)
    return CrossCorrelizer(sample_rate, shift_max, shift_steps)

class Localizer(object):
    """
        Encapsulates the process of localizing sound sources.
    """
    def __init__(self, data):
        """
            data : a dict()-like object with information for localization (or
                   the name of an npz file to load that information from).  
                   See the accompanying learn.py for how to generate that data.
        """
        if isinstance(data, str):
            data = np.load(data)

        self.angles = data['angles']
        self.decisions = data['hists'].argmax(axis=0)
        self.ccr = data['cross_correlizer'].item()

    def localize(self, sample):
        """
            Compute cross-correlation and localize the sound source.

            sample : (N x 2) array or the filename of a wav file.
        """
        if isinstance(sample, str):
            sr,sample = wavfile.read(sample)
        cc = self.ccr.cross_correlize(sample)
        best = cc.argmax()
        mle = self.decisions[best]
        return self.angles[mle]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='ccr_data', required=True)
    parser.add_argument('-i', dest='input', required=True)
    args = parser.parse_args()

    localizer = Localizer(args.ccr_data)
    print(localizer.localize(args.input))
