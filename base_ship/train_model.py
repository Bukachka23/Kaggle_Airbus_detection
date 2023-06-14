import pandas as pd
import numpy as np
import os
from PIL import Image
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Define the directory path where your data is stored
data_dir = 'actual_path'

# Load the CSV file
df = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))

# Initialize lists to store the images and masks
images = []
masks = []


# Define a function to convert RLE to mask
def rle_decode(mask_rle, shape=(768, 768)):
    # Mask RLE
    s = mask_rle.split()
    # First element is the start of the run
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # Last element is the end of the run
    starts -= 1
    # Create mask
    ends = starts + lengths
    # Create a binary mask
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # Set the non-zero pixels inside the binary mask to 1
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# Loop over the rows of the DataFrame
for idx, row in df.iterrows():
    # Get the image ID and the RLE mask
    image_id = row['ImageId']
    rle_mask = row['EncodedPixels']

    if isinstance(rle_mask, str):  # Only process images that have ship
        # Decode the RLE mask
        mask = rle_decode(rle_mask)

        # Open the image file
        image = Image.open(os.path.join(data_dir, f'{image_id}'))

        # Append the image and mask to the lists
        images.append(np.array(image))
        masks.append(mask)

    # Print a progress message every 1000 images
    if idx % 1000 == 0:
        print(f"Processed {idx} images")


# Convert lists to numpy arrays
print("Converting lists to arrays")
X = np.array(images)
Y = np.array(masks)[..., np.newaxis]


# Split the data into train and validation sets
print("Splitting data into train and validation sets")
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# Define the unet model
def unet_model(input_size=(768, 768, 1)):
    inputs = Input(input_size)

    # Downsampling path
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Upsampling path
    up1 = UpSampling2D(size=(2, 2))(pool1)
    merge1 = concatenate([conv1, up1], axis=3)
    conv2 = Conv2D(1, (1, 1), activation='sigmoid')(merge1)

    return Model(inputs=inputs, outputs=conv2)


# Define a custom metric (Dice coefficient)
def dice_coef(y_true, y_pred, smooth=1):
    # Flatten the predictions and the ground truth
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    # Add a small epsilon to avoid division by zero
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return (2. * intersection + smooth) / (union + smooth)


# Compile the model
model = unet_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[dice_coef])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('model.h5')

