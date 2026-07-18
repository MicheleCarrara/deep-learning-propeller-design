# Deep-Learning-propeller-design

_Master Thesis Project.
Mechanical Engineering, Università degli Studi di Bergamo_

## Table of contents
- [Overview](#overview)
- [Neural Network Architerature](#neural-network-architerature)
- [Results](#results)

## Overview
This project aims to reconstruct propeller geometries starting from the desired aerodynamics performance data using a Data-Driven Deep Neural Network.
Traditional propeller design requires multiple iterations between geometry generation and aerodynamic evaluation.

This project investigates whether a deep neural network can
directly infer propeller geometry from desired performance targets.

<img width="1020" height="513" alt="Screenshot 2026-07-18 103111" src="https://github.com/user-attachments/assets/cd4a0bab-ca66-4937-ae0f-f5c6a1579a2b" />

The training data is taken from apcprop.com, wich contains about 500 hundred different helics used for radio-controlled modellings and the corresponding aerodynamic performance coefficients like thrust, power and torque.
The network was built in Python using Keras API of TensorFlow library:

## Neural Network Architerature

Input layer neurons:
- Cruise velocity
- Efficiency
- Thrust
- Power
- Torque
- Peripheral Reynolds number (at tip)
- Peripheral Mach number (at tip)
- RPMs

Total of 8 neurons

Output layer neurons:
- Diameter
- Hub radius
- Pitch
- N° blades
- Airfoil section
- 15 radial chord distribution
- 11 radial twist distribution
- 12 radial thickness distribution

Total of 46 neurons

The network is composed of 3 hidden layers of 512, 256 and 128 neurons respectively
'''
model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),     
    layers.Dense(46, activation="linear")
])
'''
The *loss function* used is the Mean Squared Error (mse):
\begin{equation}
    MSE=\sum_i^N \frac{(y_{pred}-y_{true})^2}{N}
\end{equation}
Where $y_pred$ represents the geometry output predicted by the neural network and $y_true$ represents the real geometry given from apc data.
'''
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse")
'''
<img width="1176" height="718" alt="Screenshot 2026-07-18 110640" src="https://github.com/user-attachments/assets/1cbecd6e-dcde-42df-a1c6-a07eaf7832b2" />

## Results
The image below demonstrates how the loss function (mse) decreases as a function of the training eopches until it settles to a certain value.
<img width="1000" height="400" alt="Training_MSE" src="https://github.com/user-attachments/assets/54c4c150-2500-4715-8d22-e248ca69f291" />

This other image shows the relative error (in percentage) calculated for each of the first 4 output neurons of the net. A sensibility analysis was done on the architecture of the neural network, expecially on the number of hidden layers and neurons for each layers. The better results were given by a 3 hidden layer net with 512, 256 and 128 neurons respectively. As we can see from this graph the largest relative error commited by the best architerature (3 hidden layers) is about 5%, encountered at the prediction of the hub radius. A further increase in the number of layers will occur in *overfitting*, a condition where the losses during training are low, but will significantly increase in the test phase.

The output of the net was verified with CFD (Star-CCM+), as well as a low fidelity model (XROTOR).

Target performance ---> Deep Neural Network ---> Predicited Geometry ---> CFD verification



