## Introduction

With the advancement of mobile technology, comes an increase of using such devices for illegal activities. This calls for an advancement in the investigative capabilities of institutions responsible for governing private and public security. This study is a review of the performance of a simple feed-forward network of three layers of processing units[1] for classifying the screen size of a mobile device using a dataset containing eye movement and phone usage data gathered from 18 subjects using devices of varying sizes [2].

**In this project we will develop an optimal performing Neural Network by making an Evolutionary Algorithm to modify and optimize the following** -
- Number of Hidden Layers
- Number of Neurons in each Hidden Layer
- Activation Function
- Learning Rate

The project code can be found in ***Evolutionary Network Optimization.ipynb***.

## Dataset
The dataset records eye movements and web searching activity by different subjects on three different screen sizes. These are recorded as S, M and L corresponding to Small, Medium and Large screen sizes. Devices with the screen size (diagonal) less than 4 inches (e.g., Samsung Galaxy S1 and Apple iPhone 3 or 4) are recorded in the S or small-sized category. Devices with a screen of 4.5 inches (e.g., Apple iPhone 6) are classified as M or medium-sized. Finally, phablets (a portmanteau word combining the words phone and tablet) that have a screen size of over 5.4 inches (e.g., Samsung Galaxy Note 4 or Apple iPhone 6 Plus) are recorded as L or large-sized. [2]

## References

1. Gedeon, T.D. and Harris, D., 1992, June. Progressive image compression. In [Proceedings 1992] IJCNN International Joint Conference on Neural Networks (Vol. 4, pp. 403-407). IEEE. 
2. Kim, J., Thomas, P., Sankaranarayana, R., Gedeon, T. and Yoon, H.J., 2016. Understanding eye movements on mobile devices for better presentation of search results. Journal of the Association for Information Science and Technology, 67(11), pp.2607-2619.
