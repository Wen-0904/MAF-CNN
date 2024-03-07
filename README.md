# MAF-CNN
A Multimodal Attention-Fusion Convolutional Neural Network for automatic detection of sleep disorders.
The MAF-CNN model consists of four independent CNN branches, four MSA modules and a prediction module. Firstly, The time-invariant features of the multimodal signals are extracted using CNN branches. Then , the MSA modules further extract features with different scales and fuse the feature information, and finally the sleep disorder detection results are obtained by the prediction module.

These are the source code of MAF-CNN.
# Database
We evaluate our model on the Cyclic Alternating Pattern (CAP) sleep database.The CAP sleep database is a publicly available database from PhysioNet, which is a collection of 108 PSG recordings.
# Requirements
Python 3.8.13  pytorch==1.13.0  numpy==1.23.3  pandas == 1.4.1  sklearn==0.19.1

