# crosscorrelizer
Numpy/SciPy-based implementation of cross-correlation for sound-source localization

To calibrate, 

 * record sound files (wav format) with sound sources at different angles
   from your two-microphone system.  Note the angle to the sound sources in the file
   names.
 * Update training_config.yaml; in particular update the regex for extracting the 
   angle to the sound source from the sound files' names.
 * Run train.py to generate a calibration file.

That calibration file can then be used with crosscorrelizer.py to estimate the angle
to a sound source from new wav files recorded with the same two-microphone system.
