Considers models poisoned with BadNet and DFST backdoor, for CIFAR, MNIST, GTSRB datasets respy.
Uses Prophecy to extract rules to detect poisoned inputs at runtime.
Correction is done by either (1) Guessing the correct labels based on rules, (2) Masking the poison trigger in the input image.
