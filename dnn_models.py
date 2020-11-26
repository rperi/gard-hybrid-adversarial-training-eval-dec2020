from sail_dev.models import RawAudioCNN
import torch

def get_model(weights_file=None):
    if weights_file is None:
        model = RawAudioCNN(num_class=40)
    else:
        model = torch.load(weights_file)	
    return model
