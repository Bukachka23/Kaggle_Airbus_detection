import pandas as pd
import numpy as np
import os
from PIL import Image
from keras.models import load_model
from base_ship.train_model import dice_coef

# Load the trained model
model = load_model('model.h5', custom_objects={'dice_coef': dice_coef})

# Define the directory path where your data is stored
data_dir = 'actual_path'

# Load the CSV file
df_test = pd.read_csv(os.path.join(data_dir, 'test_ship_segmentations_v2.csv'))

# Initialize a list to store the images
images_test = []

# Loop over the rows of the DataFrame
for idx, row in df_test.iterrows():
    # Get the image ID
    image_id = row['ImageId']

    # Open the image file
    image = Image.open(os.path.join(data_dir, f'{image_id}'))

    # Append the image to the list
    images_test.append(np.array(image))

# Convert the list to a numpy array
X_test = np.array(images_test)

# Use the model to make predictions
predictions = model.predict(X_test)