#!/usr/bin/env python
# coding: utf-8

""" 
    This script analyzes a bunch of wav files to learn about the interaural 
    time differences for a given (binaural) system.

    The result can be used with the class Localizer from the accompanying
    script crosscorrelizer.py to localize sound sources.
"""

import re
import os.path
from collections import defaultdict,Counter
from scipy.io import wavfile
import numpy as np
import yaml

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='config', required=True)
parser.add_argument('-o', dest='outfile', required=True)
parser.add_argument('-i', dest='infiles', nargs='+', required=True)
args = parser.parse_args()

from crosscorrelizer import cross_correlizer

with open(args.config) as configfile:
    config = yaml.load(configfile)
filename_pattern = config['filename_pattern']
sample_rate = config['sample_rate']
sample_length = config['sample_length']
max_itd = config['max_itd']
max_frequency = config['max_frequency']

ccr = cross_correlizer(sample_rate, max_itd, max_frequency)

histograms = defaultdict(Counter)

hist_len = None

for infile_name in args.infiles:
    logger.info("Handling file: %s", infile_name)
    angle = int(re.match(filename_pattern, os.path.basename(infile_name)).group('angle'))
    logger.info("Angle is: %d", angle)

    sr,infile = wavfile.read(infile_name)
    assert sr == sample_rate

    num_samples = int(len(infile) / sample_length / sample_rate)
    logger.info("Number of samples: %d", num_samples)
    ccr_maxs = Counter()
    for offset in range(num_samples):
        start = offset * sample_length * sample_rate
        end   = start  + sample_length * sample_rate
        sample = infile[start:end]
        hist = ccr.cross_correlize(sample)
        ccr_maxs.update((hist.argmax(),))

        if hist_len is None:
            hist_len = len(hist)
        else:
            assert hist_len == len(hist)

    histograms[angle].update(ccr_maxs)

angles = np.array(sorted(histograms))
max_match = hist_len
hists = np.zeros((len(angles),max_match), dtype=float)

for i,a in enumerate(sorted(histograms)):
    h = histograms[a]
    for m in range(max_match):
        hists[i,m] = h[m]

hists = hists.T

hists /= hists.sum(axis=0)

np.savez(args.outfile, angles=angles, hists=hists.T, cross_correlizer = ccr)
