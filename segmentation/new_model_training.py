import segmentation_models_pytorch as smp
import torch

ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['fracture']
ACTIVATION = 'sigmoid'
DEVICE = torch.device('cuda:0')

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
).to(DEVICE)