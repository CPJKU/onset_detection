onset_detection
===============

Python implementation of the most common spectral based onset detection algorithms.

Usage
-----
`onset_program.py input.wav` processes the audio file and writes the detected onsets to a file named input.onsets.txt. This file is suitable for evaluation with the `onset_program.py` scripts which expects pairs of files with the extensions `.onsets.txt` and the corresponding ground-truth annotations ending with `.onsets`.

Please see the `-h` option to get a more detailed description.

