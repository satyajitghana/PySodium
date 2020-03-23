import seaborn as sns
from .core import GradCAM

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np
import cv2
import seaborn as sns

from sodium.utils import setup_logger

logger = setup_logger(__name__)


def get_gradcam(images, labels, model, device, target_layers):
    # move the model to device
    model.to(device)

    # set the model in evaluation mode
    model.eval()

    # get the grad cam
    gcam = GradCAM(model=model, candidate_layers=target_layers)

    # images = torch.stack(images).to(device)

    # predicted probabilities and class ids
    pred_probs, pred_ids = gcam.forward(images)

    # actual class ids
    # target_ids = torch.LongTensor(labels).view(len(images), -1).to(device)
    target_ids = labels.view(len(images), -1).to(device)

    # backward pass wrt to the actual ids
    gcam.backward(ids=target_ids)

    # we will store the layers and correspondings images activations here
    layers_region = {}

    # fetch the grad cam layers of all the images
    for target_layer in target_layers:
        logger.info(f'generating Grad-CAM for {target_layer}')

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        layers_region[target_layer] = regions

    # we are done here, remove the hooks
    gcam.remove_hook()

    return layers_region, pred_probs, pred_ids


sns.set()
plt.style.use("dark_background")


def plot_gradcam(gcam_layers, images, target_labels, predicted_labels, class_labels, denormalize, paper_cmap=False):

    images = images.cpu()
    # convert BCHW to BHWC for plotting stufffff
    images = images.permute(0, 2, 3, 1)
    target_labels = target_labels.cpu()

    fig, axs = plt.subplots(nrows=len(images), ncols=len(
        gcam_layers.keys())+1, figsize=(len(images)*3, len(gcam_layers.keys())*3))
    fig.suptitle("Grad-CAM", fontsize=16)

    for image_idx, image in enumerate(images):

        # denormalize the imaeg
        denorm_img = denormalize(image.permute(2, 0, 1)).permute(1, 2, 0)

        axs[image_idx, 0].imshow(
            (denorm_img.numpy() * 255).astype(np.uint8),  interpolation='bilinear')
        axs[image_idx, 0].set_title(
            f'predicted: {class_labels[predicted_labels[image_idx][0] ]}\nactual: {class_labels[target_labels[image_idx]] }')
        axs[image_idx, 0].axis('off')

        for layer_idx, layer_name in enumerate(gcam_layers.keys()):
            # gets H X W of the cam layer
            _layer = gcam_layers[layer_name][image_idx].cpu().numpy()[0]
            heatmap = 1 - _layer
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(
                (denorm_img.numpy() * 255).astype(np.uint8), 0.5, heatmap_img, 0.5, 0)

            axs[image_idx, layer_idx +
                1].imshow(superimposed_img, interpolation='bilinear')
            axs[image_idx, layer_idx+1].set_title(f'layer: {layer_name}')
            axs[image_idx, layer_idx+1].axis('off')

    plt.tight_layout()
    plt.show()
