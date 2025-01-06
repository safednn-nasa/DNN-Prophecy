Corresponds to work done in paper, 	
Muhammad Usman, Divya Gopinath, Youcheng Sun, Corina S. Pasareanu:
Rule-Based Runtime Mitigation Against Poison Attacks on Neural Networks. RV 2022: 67-84
https://pure.manchester.ac.uk/ws/files/225542785/AntidoteRT_RV.pdf

Considers models poisoned with BadNet and DFST backdoor, for CIFAR, MNIST, GTSRB datasets respy.
Uses Prophecy to extract rules to detect poisoned inputs at runtime.
Correction is done by either (1) Guessing the correct labels based on rules, (2) Masking the poison trigger in the input image.
