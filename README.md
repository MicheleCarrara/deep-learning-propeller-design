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
model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),     
    layers.Dense(46, activation="linear")
])


The loss *function* used is the Mean Squared Error (MSE)

<img width="1176" height="718" alt="Screenshot 2026-07-18 110640" src="https://github.com/user-attachments/assets/1cbecd6e-dcde-42df-a1c6-a07eaf7832b2" />

## Results


The output of the net was verified with CFD (Star-CCM+), as well as a low fidelity model (XROTOR).

Target performance ---> Deep Neural Network ---> Predicited Geometry ---> CFD verification



