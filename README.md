# Ship Segmentation using Simplified U-Net

This project is aimed at detecting and segmenting ships in satellite images using a simplified version of the U-Net architecture. The main objective is to implement a simplified U-Net model that can accurately predict the presence of ships in the images and generate a corresponding segmentation mask.

## Project Structure

- The project consists of two primary Python scripts:

- train_model.py: This script is responsible for several tasks related to the training of the model. Specifically:

- It loads the training data, which consists of satellite images and corresponding ship masks encoded in RLE (Run Length Encoding) format.
- It preprocesses the data, including decoding the RLE masks and converting the images and masks into numpy arrays.
- It defines the architecture of the simplified U-Net model. This model includes a single convolution and pooling layer in the downsampling path, and a single upsampling and convolution layer in the upsampling path.
- It compiles the model, using the Adam optimizer and binary cross-entropy as the loss function. A custom Dice coefficient function is used as a metric.
- It trains the model using the prepared training data and saves the trained model to a file named model.h5.
inference.py: This script is responsible for loading the trained model and using it to make predictions on the test data. It uses the Keras load_model function to load the model from the model.h5 file, and then uses the model's predict method to generate segmentation masks for the test images.

## Requirements
This project requires Python 3.6+ and the following Python libraries installed:

- Numpy: A library for numerical computations in Python.
- Pandas: A data manipulation and analysis library.
- Pillow: The Python Imaging Library, used for opening, manipulating, and saving many different image file formats.
- TensorFlow: A machine learning platform, used as the backend for Keras.
- Keras: A high-level neural networks API, used for building and training the model.
- scikit-learn: A machine learning library, used here for splitting the dataset into training and validation sets.

- You can install these dependencies by running the following command in your terminal:
pip install -r requirements.txt

- To train the model, you need to run the train_model.py script. This can be done with the following command:
python train_model.py
- This will train the model using your training data and save the trained model to a file named model.h5.

- After you have trained the model, you can use it to make predictions on your test data. This can be done by running the inference.py script 

- with the following command:
python inference.py
- This will load the trained model from the model.h5 file and use it to generate segmentation masks for the test images.

## Data
- The data for this project should consist of satellite images and associated ship segmentation masks. These are loaded from a CSV file in the train_model.py script. The CSV file should have two columns: 'ImageId' and 'EncodedPixels'. The 'ImageId' column should contain the filenames of the images, and the 'EncodedPixels' column should contain the RLE-encoded ship segmentation masks.

- The test data is loaded in a similar way in the inference.py script, from a separate CSV file.

## Model
- The model used in this project is a simplified version of the U-Net architecture, which is a type of convolutional neural network that is widely used for image segmentation tasks. The U-Net architecture consistsof a downsampling path that captures the context in the image, and an upsampling path that enables precise localization, making it ideal for tasks like semantic segmentation.

- In this simplified U-Net model, the downsampling path consists of a single convolution layer followed by a max pooling layer. The convolution layer applies a set of filters to the input image, effectively learning image features at different levels of abstraction. The max pooling layer then reduces the spatial dimensions of the input, making the model more computationally efficient.

- The upsampling path of the model consists of an upsampling layer followed by a concatenation and then another convolution layer. The upsampling layer increases the spatial dimensions of the input, allowing the model to make predictions at the original image resolution. The concatenation merges the outputs of the upsampling and downsampling paths, allowing the model to use information from both paths when making predictions. The final convolution layer applies a sigmoid activation function, producing the final segmentation mask.

- The model is compiled with the Adam optimizer, a popular choice for deep learning models due to its efficiency and low memory requirements. The loss function is binary cross-entropy, which is appropriate for binary classification tasks like ship/no-ship segmentation. Finally, the Dice coefficient is used as a metric during training. The Dice coefficient is a common metric for image segmentation tasks, as it provides a measure of overlap between the predicted and ground truth segmentation masks.

## Note
- This is a simplified version of the U-Net model and might not give the best results for complex real-world ship segmentation tasks. For more challenging tasks, consider using a full U-Net model or other advanced segmentation models, potentially with more layers or additional regularization techniques.

