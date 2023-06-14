import matplotlib.pyplot as plt
import numpy as np

# Assuming test_images is your array of test images and pred_masks is your array of predicted masks
for i in range(len(test_images)):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax[0].imshow(test_images[i], cmap='gray')
    ax[0].set_title('Original Image')

    # Show mask (predicted by the model)
    ax[1].imshow(np.squeeze(pred_masks[i]), cmap='gray')
    ax[1].set_title('Predicted Mask')

    plt.show()
