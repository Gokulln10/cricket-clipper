"""Audio-event classifier: training + inference for cricket events.

This package trains a small CNN on log-mel spectrograms to recognise events
such as ``four``, ``six`` and ``wicket`` from a clip's audio. The trained model
plugs into the main clipper as a high-precision detector (see
``clipper/classifier.py``).

Requires the optional ML dependencies:

    pip install -r requirements-ml.txt
"""
