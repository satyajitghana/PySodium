from .core import GradCAM

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


def get_gradcam(images, labels, model, device, target_layers):
    # move the model to device
    model.to(device)

    # set the model in evaluation mode
    model.eval()

    gcam = GradCAM(model=model, candidate_layers=target_layers)

    images = torch.stack(images).to(model.device)

    probs, ids = gcam.forward(images)

    ids_ = torch.LongTensor(labels).view(len(images), -1).to(device)

    gcam.backward(ids=ids_)

    layers_region = []

    for target_layer in target_layers:
        print(f'Generating Grad-CAM for {target_layer}')

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        layers_region.extend(regions)

    gcam.remove_hook()

    return layers_region, probs, ids


def plot_gradcam(gcam_layers, images, labels, probs, denormalize):

    plt.axis('off')

    for idx, image in enumerate(images):
        denorm_img = np.uint8(255 * denormalize(image.view(3, 32, 32)))
        plt.imshow(denorm_img)
        plt.axis('off')

        for jdx, gcam_layer in enumerate(gcam_layers):
            heatmap = 1 - gcam_layer[idx].cpu().numpy()[0]

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.resize(cv2.addWeighted(
                img, 0.5, heatmap, 0.5, 0), (128, 128))
            plt.imshow(superimposed_img)

    plt.show()
