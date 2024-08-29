# Automating Port Operations

Course end project for Deep Learning. Uses Convolutional Neural Network (CNN) & transfer models with MobileNetV2 that would improve model accuracy & precision compared to a simpler model.

**NOTICE: Please unzip the .zip file before running the program, or otherwise, the program cannot get access to the directories and therefore not work.**

## The Project

Marina Pier Inc. is leveraging technology to automate their operations on the San Francisco port.

The company’s management has set out to build a bias-free/ corruption-free automatic system that reports & avoids faulty situations caused by human error. Examples of human error include misclassifying the correct type of boat. The type of boat that enters the port region is as follows.

- Buoy
- Cruise_ship
- Ferry_boat
- Freight_boar
- Gondola
- Inflatable_boat
- Kayak
- Paper_boat
- Sailboat

Marina Pier wants to use Deep Learning techniques to build an automatic reporting system that recognizes the boat. The company is also looking to use a transfer learning approach of any lightweight pre-trained model in order to deploy in mobile devices.

As a deep learning engineer, your task is to:

1. Build a CNN network to classify the boat.

2. Build a lightweight model with the aim of deploying the solution on a mobile device using transfer learning. You can use any lightweight pre-trained model as the initial (first) layer. MobileNetV2 is a popular lightweight pre-trained model built using Keras API

### Perform the following steps:

1. Build a CNN network to classify the boat.
    1. Split the dataset into train and test in the ratio 80:20, with shuffle and random state=43. 
    2. Use tf.keras.preprocessing.image_dataset_from_directory to load the train and test datasets. This function also supports data normalization. (Hint: image_scale=1./255).
    3. Load train, validation and test dataset in batches of 32 using the function initialized in the above step. 
    4. Build a CNN network using Keras with the following layers
        * Cov2D with 32 filters, kernel size 3,3, and activation relu, followed by MaxPool2D
        * Cov2D with 32 filters, kernel size 3,3, and activation relu, followed by MaxPool2D
        * GLobalAveragePooling2D layer
        * Dense layer with 128 neurons and activation relu
        * Dense layer with 128 neurons and activation relu
        * Dense layer with 9 neurons and activation softmax.
    5. Compile the model with Adam optimizer, categorical_crossentropy loss, and with metrics accuracy, precision, and recall.
    6. rain the model for 20 epochs and plot training loss and accuracy against epochs.
    7. Evaluate the model on test images and print the test loss and accuracy.
    8. Plot heatmap of the confusion matrix and print classification report.

2. Build a lightweight model with the aim of deploying the solution on a mobile device using transfer learning. You can use any lightweight pre-trained model as the initial (first) layer. MobileNetV2 is a popular lightweight pre-trained model built using Keras API. 
    1. Split the dataset into train and test datasets in the ration 70:30, with shuffle and random state=1.
    2. Use tf.keras.preprocessing.image_dataset_from_directory to load the train and test datasets. This function also supports data normalization. (Hint: Image_scale=1./255).
    3. Load train, validation and test datasets in batches of 32 using the function initialized in the above step.
    4. Build a CNN network using Keras with the following layers. 
        * Load MobileNetV2 - Light Model as the first layer (Hint: Keras API Doc)
        * GLobalAveragePooling2D layer
        * Dropout(0.2)
        * Dense layer with 256 neurons and activation relu
        * BatchNormalization layer
        * Dropout(0.1)
        * Dense layer with 128 neurons and activation relu
        * BatchNormalization layer
        * Dropout(0.1)
        * Dense layer with 9 neurons and activation softmax
    5. Compile the model with Adam optimizer, categorical_crossentropy loss, and metrics accuracy, Precision, and Recall.
    6. Train the model for 50 epochs and Early stopping while monitoring validation loss.
    7. Evaluate the model on test images and print the test loss and accuracy.
    8. Plot Train loss Vs Validation loss and Train accuracy Vs Validation accuracy.
    
3. Compare the results of both models built in steps 1 and 2 and state your observations.