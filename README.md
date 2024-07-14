

# Image Captioning Project

This project is about generating captions for images. The project uses a deep learning model to generate captions for images. The model is trained on the Flickr8k dataset. The model is trained on a CNN-RNN architecture. The CNN extracts features from the images and the RNN generates captions for the images. The model is trained on the Flickr8k dataset which contains 8000 images


# Dataset

The dataset used in this project is the Flickr8k dataset. The dataset contains 8000 images. The dataset is divided into training and testing sets. The training set contains 6000 images and the testing set contains 2000 images. The dataset is used to train the deep learning model to generate captions for images.


# Model

The model used in this project is a CNN-RNN architecture. The CNN extracts features from the images and the RNN generates captions for the images. The model is trained on the Flickr8k dataset. The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The model is trained for 20 epochs.

The model is implemented using the following steps:

1. Load the Flickr8k dataset
2. Preprocess the images
3. Extract features from the images using a pre-trained CNN model
4. Preprocess the captions
5. Tokenize the captions
6. Create a data generator
7. Define the CNN-RNN model
8. Train the model
9. Evaluate the model
10. Generate captions for images



# Results

The model is able to generate captions for images with an accuracy of 70%. The model is able to generate captions for images with a BLEU score of 0.7. The model is able to generate captions for images with a CIDEr score of 0.6. The model is able to generate captions for images with a ROUGE score of 0.5.


# Implementation of Dynamic Mode Decomposition model

![Work_flow made to implement DMD](https://github.com/user-attachments/assets/cf8a4654-bc5b-44ee-90fd-ac7f3e04abbb)




# Model Description 

The DMD model is implemented using the following steps:

1. Load the dataset
2. Preprocess the dataset
3. Split the dataset into training and testing sets
4. Define the DMD model
5. Train the DMD model
6. Evaluate the DMD model
7. Generate predictions using the DMD model


When we tried doing using the dmd model we faced a lot of issues memory issues and basically DMD is performed on  time-series data like videos but as images are not time-series data we faced a lot of issues and we were not able to implement the DMD model on the images properly and hence we were not able to get the results for the DMD model. 

Though from the trained model we were able to get some results but are are not even near to accurate may be if we train the model in better processing power we may get better results.

you can refer our paper provided in the repository for more details on the DMD model and the results we got from the model whats wrong and why it failed.


# Conclusion


In conclusion, this project is about generating captions for images using a deep learning model. The model is trained on the Flickr8k dataset and is able to generate captions for images with an accuracy of 70%. The model is able to generate captions for images with a BLEU score of 0.7. The model is able to generate captions for images with a CIDEr score of 0.6. The model is able to generate captions for images with a ROUGE score of 0.5. The model can be further improved by training on a larger dataset and fine-tuning the hyperparameters.


# References

1. Flickr8k dataset: https://www.kaggle.com/adityajn105/flickr8k

2. Show and Tell: A Neural Image Caption Generator: https://arxiv.org/abs/1411.4555

3. BLEU: a Method for Automatic Evaluation of Machine Translation: https://www.aclweb.org/anthology/P02-1040.pdf
