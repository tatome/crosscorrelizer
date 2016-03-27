#!/usr/bin/env python
# coding: utf-8

import re
import os.path
from collections import defaultdict,Counter
from scipy.io import wavfile
import numpy as np
import scipy.stats
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

angles = []
ccrs   = []

for infile_name in args.infiles:
    logger.debug("Handling file: %s", infile_name)
    angle = int(re.match(filename_pattern, os.path.basename(infile_name)).group('angle'))
    logger.debug("Angle is: %d", angle)

    sr,infile = wavfile.read(infile_name)
    assert sr == sample_rate

    num_samples = int(len(infile) / sample_length / sample_rate)
    logger.debug("Number of samples: %d", num_samples)
    ccr_maxs = Counter()
    for offset in range(num_samples):
        start = offset * sample_length * sample_rate
        end   = start  + sample_length * sample_rate
        sample = infile[start:end]
        hist = ccr.cross_correlize(sample)
        angles.append(angle)
        ccrs.append(hist.argmax())

np.savez(args.outfile, angles = angles, ccrs = ccrs)
logger.info("Pearson's rho: %.4f (%.5f)", *scipy.stats.spearmanr(angles, ccrs))
