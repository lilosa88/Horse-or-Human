# Horse-or-Human

# Objective

- This project belongs to [kaggle's competitions](https://www.kaggle.com/sanikamal/horses-or-humans-dataset) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

  Specifically this project is part of the first course in this specialization. 

- Horses or Humans is a dataset of 300×300 images, created by Laurence Moroney, that is licensed CC-By-2.0 for anybody to use in learning or testing computer vision algorithms.

- The objective of this study is to correctly identify if the image is a horse or a human.

# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, seaborn, matplotlib, tensorflow, keras.

# Data description  

- The set contains 500 rendered images of various species of horse in various poses in various locations. It also contains 527 rendered images of humans in various poses and locations. Emphasis has been taken to ensure diversity of humans, and to that end there are both men and women as well as Asian, Black, South Asian and Caucasians present in the training set. The validation set adds 6 different figures of different gender, race and pose to ensure breadth of data.

- The images looks like:
  <p align="center">
   <img src="https://github.com/lilosa88/Horse-or-Human/blob/main/Images/Captura%20de%20Pantalla%202021-05-18%20a%20la(s)%2017.17.26.png" width="720" height="480">
  </p> 
  
# Feature engineering

- The Fashion MNIST data is available directly in the tf.keras datasets API. Using load_data we get two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.
 
- The values in the number are between 0 and 255. Since we will train a neural network, we need that all values are between 0 and 1. Therefore, we normalize dividing by 255.

- We reshape the images (only for the second model), following training_images.reshape(60000, 28, 28, 1) and test_images.reshape(10000, 28, 28, 1)


# Neural Network model

### First model: Simple Neural Network

- This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:
  - One flatten layer: It turns the images into a 1 dimensional set.
  - Three Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer         consisted in 1024 neurons with relu as an activation function. The second, have 128 neurons and the same activation function. Finally, the thrird had 10 neurons     and softmax as an activation function. Indeed, the number of neurons in the last layer should match the number of classes you are classifying for. In this case     it's the digits 0-9, so there are 10 of them.

- We built this model using Adam optimizer and sparse_categorical_crossentropy as loss function.

- We obtained Accuracy 0.9299 for the train data and Accuracy 0.8923 for the validation data.

### Second model: Neural Network with convolution and pooling

- This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:
  - One Convolution layer with a MaxPooling layer which is then designed to compress the image, while maintaining the content of the features that were                 highlighted by the convlution. By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image.
  - One flatten layer: It turns the images into a 1 dimensional set.
  - Three Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer         consisted in 1024 neurons with relu as an activation function. The second, have 128 neurons and the same activation function. Finally, the thrird had 10 neurons     and softmax as an activation function. Indeed, the number of neurons in the last layer should match the number of classes you are classifying for. In this case     it's the digits 0-9, so there are 10 of them.

- We built this model using Adam optimizer and sparse_categorical_crossentropy as loss function.

- We obtained Accuracy 0.9953 for the train data and Accuracy 0.9147 for the validation data.
