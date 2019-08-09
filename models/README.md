# Pretrained Models

### Polish lanugage
The polish model weights are available [here](https://storage.googleapis.com/deepspeech-keras-weights/weights.hdf5).
Please normalized each feature channels based on `features-stats.txt`.
The samples are clipped to 7 words long (otherwise the LSTM could explode).
This is an initial version. This model achieves 16.4% WER on hold out [Clarin Dataset](https://github.com/danijel3/ClarinStudioKaldi). 
