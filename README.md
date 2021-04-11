# Submission for MIDAS Task 2

- [Submission for MIDAS Task 2](#submission-for-midas-task-2)
  - [Summary](#summary)
  - [Short Info on the methods used](#short-info-on-the-methods-used)
  - [Declaration](#declaration)
  - [References](#references)

## Summary

I've used [Google Colab](https://colab.research.google.com/) for performing the task.

I have attempted the Task 2 and the Part 1 of the task is in the [`Task2/midastask2part1.ipynb`](Task2/midastask2part1.ipynb) notebook and Parts 2 and 3 are in [`Task2/midastask2part2-3.ipynb`](Task2/midastask2part2-3.ipynb) notebook.

The three parts of the tasks have been split into two Jupyter Notebooks, which can be uploaded to Google Colab and run with a GPU instantly without any prerequisites or other installation required.

I've added all implementation choices, experiments, observations, results and details in the notebooks along with the code.

## Short Info on the methods used

Tensorflow and Keras have been used to complete the tasks. I've built Neural Networks based on my notes prepared using ref. [16] and online resource, referenced below.

Starting with the original LeNet in part 1, I've used various methods to improve the model, which I have then used for completing the following parts of the task.

## Declaration

I hereby declare that all the work done for this task is my own and I have provided due references which I have used for performing these experiments.

## References

-   [1] [EMNIST handwritten character recognition with Deep Learning](https://medium.com/@mrkardostamas/emnist-handwritten-character-recognition-with-deep-learning-b5d61ac1aab7)
-   [2] [How to choose CNN Architecture MNIST](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist)
-   [3] [Swish Vs Mish: Latest Activation Functions](https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/)
-   [4] [Mish Class Definition in Keras](https://gist.github.com/digantamisra98/35ca0ec94ebefb99af6f444922fa52cd)
-   [5] [Mnist_EfficientNet Kaggle Notebook](https://www.kaggle.com/ateplyuk/mnist-efficientnet)
-   [6] [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#training-a-model-from-scratch)
-   [7] [Keras EfficientNet Implementation Source Code](https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/efficientnet.py#L517-L540)
-   [8] [Keras ModelCheckpoint Documentaion](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
-   [9] [Keras EarlyStopping Documentaion](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
-   [10] [Keras ImageDataGenerator Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
-   [11] [Keras EfficientNetB0 Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)
-   [12] [Effect of Batch Size on Neural Net Training](https://medium.com/deep-learning-experiments/effect-of-batch-size-on-neural-net-training-c5ae8516e57)
-   [13] [How does temperature affect softmax in machine learning?](http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html)
-   [14] [Temperatured softmax · Issue #3092 · keras-team/keras](https://github.com/keras-team/keras/issues/3092)
-   [15] [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf)
-   [16] [Notes from Deeplearning.ai's Deep Learning Speciaization](https://www.coursera.org/specializations/deep-learning)
