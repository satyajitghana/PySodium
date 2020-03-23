from .core import GradCAM

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np
import cv2


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
    target_ids = torch.LongTensor(labels).view(len(images), -1).to(device)

    # backward pass wrt to the actual ids
    gcam.backward(ids=target_ids)

    # we will store the layers and correspondings images activations here
    layers_region = {}

    # fetch the grad cam layers of all the images
    for target_layer in target_layers:
        print(f'Generating Grad-CAM for {target_layer}')

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        layers_region.setdefault(target_layer, []).append(regions)

    # we are done here, remove the hooks
    gcam.remove_hook()

    return layers_region, pred_probs, pred_ids


def plot_gradcam(gcam_layers, images, target_labels, predicted_labels, denormalize, paper_cmap=False):

    image_shape = images.shape[1:]

    plt.axis('off')

    fig, axs = plt.subplots(nrows=len(images), ncols=len(gcam_layers.keys()))

    fig.suptitle('Grad-CAM')

    for image_idx, image in images:

        axs[image_idx, 0].imshow(image)

        for layer_idx, layer_name in enumerate(gcam_layers.keys()):

            # gets H X W of the cam layer
            _layer = gcam_layers[layer_name][image_idx].cpu().numpy()[0]
            cmap = cm.jet_r(_layer)[..., :3] * 255.0

            gcam = (cmap.astype(np.float) +
                    images[image_idx].astype(np.float))/2
            gcam = np.uint8(gcam)

            axs[image_idx, layer_idx+1].imshow(gcam)

    plt.show()

    # for idx, image in enumerate(images):
    #     denorm_img = np.uint8(255 * denormalize(image.view(image_shape)))
    #     plt.axis('off')
    #     plt.imshow(denorm_img, interpolation='bilinear')
    #     plt.show()

    #     for jdx in range(4):
    #         heatmap = 1 - gcam_layers[len(images)*jdx + idx].cpu().numpy()[0]

    #         heatmap = np.uint8(255 * heatmap)
    #         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #         # print(heatmap.shape)
    #         # print(denorm_img.shape)
    #         superimposed_img = cv2.resize(cv2.addWeighted(
    #             denorm_img, 0.5, heatmap, 0.5, 0), (128, 128))
    #         plt.axis('off')
    #         plt.imshow(superimposed_img, interpolation='bilinear')
    #         plt.show()

    # plt.show()
