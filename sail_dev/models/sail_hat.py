"""
CNN model for raw audio classification
The model extracts mel-spectrogram online, and trains with it.
The gradient flows smoothly to the raw time domain audio,
so that adversarial samples can be generated in the time domain.
Model contributed by: USC SAIL (sail.usc.edu)
"""
from art.classifiers import PyTorchClassifier
import numpy as np
import torch

from armory.data.utils import maybe_download_weights_from_s3

import dnn_models

#TODO: eval time full len testing should be included
#Now it trains and evals with 3s segments
WINDOW_LENGTH=48000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#adpted from sincnet.py
def preprocessing_fn(batch):
    """
    Standardize, then normalize sound clips
    """
    processed_batch = []
    for clip in batch:
        signal = clip.astype(np.float64)
        # Signal normalization
        signal = signal / np.max(np.abs(signal))

        # get pseudorandom chunk of fixed length (from SincNet's create_batches_rnd)
        signal_length = len(signal)

        if signal_length < WINDOW_LENGTH:
           signal = np.concatenate((signal, np.zeros(WINDOW_LENGTH-signal_length)))
        else:
            np.random.seed(signal_length)
            signal_start = np.random.randint(0, signal_length-WINDOW_LENGTH)
            signal_stop = signal_start + WINDOW_LENGTH
            signal = signal[signal_start:signal_stop]

        processed_batch.append(signal)

    return np.array(processed_batch)

def sail(weights_file=None):
    pretrained = weights_file is not None
    filepath = None
    if pretrained:
        filepath = maybe_download_weights_from_s3(weights_file)
    sailNet = dnn_models.get_model(weights_file=filepath)
    
    if pretrained:
        sailNet.eval()
    else:
        sailNet.train()

    return sailNet


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = sail(weights_file=weights_file)
    model.to(DEVICE)
    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            model.parameters(), lr=model_kwargs['lr'], betas=(.5, .999)
        ),
        input_shape=(WINDOW_LENGTH,),
        nb_classes=40,
    )
    return wrapped_model
