# RU-Net
Towards a Predictor for CO2 Plume Migration using Deep Neural Networks

# Abstract
This paper demonstrates a deep neural network approach for predicting carbon dioxide (CO2) plume migration from an injection well in heterogeneous formations with high computational efficiency. 
With the data generation and training procedures proposed in this paper, we show that the deep neural network model can generate predictions of CO2 plume migration that are as accurate as traditional numerical simulation, given input variables of a permeability field, an injection duration, injection rate, and injection location. The neural network model can deal with permeability fields that have high degrees of heterogeneity. Unlike previous studies which did not consider the effect of buoyancy, here we also show that the neural network model can learn the consequences of the interplay of gravity, viscous, and capillary forces, which is critically important for predicting CO2 plume migration.
The neural network model has an excellent ability to generalize within the training data ranges and to a limited extent, the ability to extrapolate beyond the training data ranges. 
To improve the prediction accuracy when the neural network model needs to extrapolate  to situations or parameters not contained in the training set, we propose a transfer learning (fine-tuning) procedure that can quickly teach the trained neural network model new information without going through massive data collection and retraining. 
With the approaches described in this paper, we have demonstrated many of the building blocks required for developing a general-purpose neural network for predicting CO2 plume migration away from an injection well.

# Requirements
Python 3.6<br>
Tensorflow-gpu 1.12.0<br>
Keras 2.2.4<br>
Numpy 1.16.1<br>