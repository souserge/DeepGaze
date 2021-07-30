
import torch
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp

def postprocess_output(log_proba):
    smap = log_proba[0, 0, :, :]
    smap = smap.exp()
    smap = torch.squeeze(smap)
    smap = smap.detach().numpy()
    smap = (smap / np.amax(smap) * 255).astype(np.uint8)
    return smap

def preprocess_input(image, centerbias_template=np.zeros((1024, 1024))):
    # remove alpha channel
    image = image[:, :, :3]

    # rescale to match image size
    centerbias = zoom(
        centerbias_template, 
        (
            image.shape[0]/centerbias_template.shape[0], 
            image.shape[1]/centerbias_template.shape[1]
        ), 
        order=0, 
        mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor([image.transpose(2, 0, 1)])
    centerbias_tensor = torch.tensor([centerbias])

    return image_tensor, centerbias_tensor
