import abc
import logging
import numpy as np
import torch
import torch.nn.functional as F
import optuna
from abc import ABC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import permutation_test_score, train_test_split
from sklearn.metrics import accuracy_score
import sys
sys.path.append("..")
from explanations.concept import ConceptExplainer
from .decision_tree import *


class DTExplainer(ConceptExplainer, ABC):
    def __init__(self, device: torch.device, batch_size: int = 50):
        super(DTExplainer, self).__init__(device, batch_size)

    def fit(self, concept_reps: np.ndarray, concept_labels: np.ndarray) -> None:
        """
        Fit the concept classifier to the dataset (latent_reps, concept_labels)
        Args:
            kernel: kernel function
            latent_reps: latent representations of the examples illustrating the concept
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
        """
        super(DTExplainer, self).fit(concept_reps, concept_labels)
        self.classifier = get_tree(concept_reps, concept_labels)
        self.invariants = get_all_invariants_vals(self.classifier)

    def validate(self, latent_reps: np.ndarray, concept_labels: np.ndarray, n:int) -> np.ndarray:
        """
        Validate the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
            concept_labels: labels indicating the presence (1) or absence (0) of the concept
            n: number of invariants to be used for validation
        Returns:
            None
        """
        # print(f'True: {len(self.invariants[True][:n])}')
        # print(f'False: {len(self.invariants[False][:n])}')
        acc_t, rc_t = validate_n_invariants(self.invariants[True][:n], True, latent_reps, concept_labels)
        acc_f, rc_f = validate_n_invariants(self.invariants[False][:n], False, latent_reps, concept_labels)
        return acc_t, rc_t, acc_f, rc_f

    def predict(self, latent_reps: np.ndarray) -> np.ndarray:
        """
        Predicts the presence or absence of the concept for the latent representations
        Args:
            latent_reps: representations of the test examples
        Returns:
            concepts labels indicating the presence (1) or absence (0) of the concept
        """
        raise NotImplementedError

    def concept_importance(
        self,
        latent_reps: np.ndarray,
        labels: torch.Tensor = None,
        num_classes: int = None,
        rep_to_output: callable = None,
    ) -> np.ndarray:
        """
        Predicts the relevance of a concept for the latent representations
        Args:
            latent_reps: representations of the test examples
            labels: the labels associated to the representations one-hot encoded
            num_classes: the number of classes
            rep_to_output: black-box mapping the representation space to the output space
        Returns:
            concepts scores for each example
        """
        raise NotImplementedError

    def permutation_test(
        self,
        concept_reps: np.ndarray,
        concept_labels: np.ndarray,
        n_perm: int = 100,
        n_jobs: int = -1,
    ) -> float:
        """
        Computes the p-value of the concept-label permutation test
        Args:
            concept_labels: concept labels indicating the presence (1) or absence (0) of the concept
            concept_reps: representation of the examples
            n_perm: number of permutations
            n_jobs: number of jobs running in parallel

        Returns:
            p-value of the statistical significance test
        """
        raise NotImplementedError