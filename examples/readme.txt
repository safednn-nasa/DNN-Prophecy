Example notebooks demonstrating the use of Prophecy for classification and regression models. 

1. Prophecy_Tool_MNIST.ipynb: Demonstrates the use of Prophecy to extract different types of rules for a CNN for MNIST. It also demonstrates the use of the Marabou tool to prove rules and use of attribution to determine input pixels impacting a rule.

2. ACASX_ProphecyTool.ipynb: Demostrates use of Prophecy to extract rules for the ACASXU (Airborne Collision Avoidance System for Unmanned Aircraft) model https://arxiv.org/pdf/2011.05174. Demonstrates use of Marabou to prove the rules.

3. GTSRB_ProphecyTool.ipynb: Demostrates use of Prophecy to extract rules for a model for GTSRB (German Traffic Sign Recognition Benchmark) https://paperswithcode.com/dataset/gtsrb.

4. Airfoil_Self_Noise_ProphecyTool.ipynb: Regression model to prediction of acoustic noise generation from airfoil: 
5D input:
Frequency [Hz]
Airfoil angle of attack [deg]
Chord length [m]
Free-stream velocity [m/s]
Suction side displacement thickness [m]
1D output:
Scaled sound pressure level at given frequency [dB]

Extracts rules and uses Marabou to generate proofs.

