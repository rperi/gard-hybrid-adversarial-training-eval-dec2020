from sail_dev.models import RawAudioCNN
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(weights_file=None):
    if weights_file is None:
        model = RawAudioCNN(num_class=40)
    else:
        model = torch.load(weights_file, map_location=device)
    return model
