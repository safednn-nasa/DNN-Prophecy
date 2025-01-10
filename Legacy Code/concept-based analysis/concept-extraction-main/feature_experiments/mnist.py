import itertools
import logging
import argparse
import torch
import numpy as np
import os
import pandas as pd
from pathlib import Path
from models.mnist import ClassifierMnist, init_trainer, get_dataloader
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from utils.dataset import generate_mnist_concept_dataset
from utils.plot import (
    plot_concept_disentanglement,
    plot_concept_accuracy,
    plot_global_explanation,
    plot_grayscale_saliency,
    plot_attribution_correlation,
    plot_kernel_sensitivity,
    plot_concept_size_impact,
    plot_tcar_inter_concepts,
)
from explanations.concept import CAR, CAV
from feature_mining.decision_tree_concept import DTExplainer
from feature_mining.entanglement import *
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
from typing import List, Tuple, Dict
from utils.robustness import Attacker

concept_to_class = {
    "Loop": [0, 6, 8, 9],
    "Vertical Line": [1, 4, 7],
    "Horizontal Line": [4, 5, 7],
    "Curvature": [0, 2, 3, 5, 6, 8, 9],
    "Nonsense": [],
}

def train_mnist_model(
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    data_dir: Path = Path.cwd() / "data/mnist",
) -> None:
    logging.info("Fitting MNIST classifier")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = ClassifierMnist(latent_dim, model_name).to(device)
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    model.fit(device, train_loader, test_loader, model_dir)


def concept_disentanglement(
    random_seeds: List[int],
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/concept_disentanglement",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    module_names = []
    dise_train = []
    dise_test = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 2000, random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_test"
        )
        X_test, y_test = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, False, 500, random_seed
        )
        model(torch.from_numpy(X_test).to(device))
        remove_all_hooks(handler_test_dic)

        if len(module_names) == 0:
            module_names = list(module_dic.keys())

        concept_e_train = []
        concept_e_test = []
        for module_name in module_dic:
            logging.info(f"Calculating disentanglement for {module_name}")
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
            H_test = get_saved_representations(hook_name, representation_dir)
            
            # Calculate disentanglement
            e_train = cal_disentanglement_assort(H_train, y_train)
            e_test = cal_disentanglement_assort(H_test, y_test)
            logging.info(f"{concept_name} {module_name} disentanglement for train set: {e_train}")
            logging.info(f"{concept_name} {module_name} disentanglement for test set: {e_test}")
            concept_e_train.append(e_train)
            concept_e_test.append(e_test)
            
        dise_train.append(concept_e_train)
        dise_test.append(concept_e_test)

        # Remove saved representations to save space
        for module_name in module_dic:
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            if os.path.exists(representation_dir/f'{hook_name}.npy'):
                os.remove(representation_dir/f'{hook_name}.npy')
            hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
            if os.path.exists(representation_dir/f'{hook_name}.npy'):
                os.remove(representation_dir/f'{hook_name}.npy')

    dise_train_df = pd.DataFrame(
        dise_train,
        columns=module_names,
        index=list(concept_to_class.keys())
    ).T
    dise_test_df = pd.DataFrame(
        dise_test,
        columns=module_names,
        index=list(concept_to_class.keys())
    ).T
    csv_train_path = save_dir / "metrics_train.csv"
    csv_test_path = save_dir / "metrics_test.csv"
    dise_train_df.to_csv(csv_train_path, header=True, mode="w", index=True)
    dise_test_df.to_csv(csv_test_path, header=True, mode="w", index=True)
    
    if plot:
        plot_concept_disentanglement(save_dir, "mnist", train=True)
        plot_concept_disentanglement(save_dir, "mnist", train=False)


def concept_accuracy(
    random_seeds: List[int],
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist/concept_accuracy",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist/",
    model_name: str = "model",
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model_dir = model_dir / model_name
    model = ClassifierMnist(latent_dim, model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 2000, random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_test"
        )
        X_test, y_test = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, False, 500, random_seed
        )
        model(torch.from_numpy(X_test).to(device))
        remove_all_hooks(handler_test_dic)
        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Fitting concept classifiers for {module_name}")
            car = CAR(device)
            cav = CAV(device)
            dt = DTExplainer(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            car.fit(H_train, y_train)
            cav.fit(H_train, y_train)
            dt.fit(H_train, y_train)
            hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
            H_test = get_saved_representations(hook_name, representation_dir)
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAR",
                    accuracy_score(y_train, car.predict(H_train)),
                    accuracy_score(y_test, car.predict(H_test)),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAV",
                    accuracy_score(y_train, cav.predict(H_train)),
                    accuracy_score(y_test, cav.predict(H_test)),
                ]
            )
            acc_t_train, rc_t_train, acc_f_train, rc_f_train = dt.validate(H_train, y_train, 10)
            acc_t_test, rc_t_test, acc_f_test, rc_f_test = dt.validate(H_test, y_test, 10)
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "DT Positive Precision",
                    acc_t_train,
                    acc_t_test,
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "DT Negative Precision",
                    acc_f_train,
                    acc_f_test,
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "DT Positive Recall",
                    rc_t_train,
                    rc_t_test,
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "DT Negative Recall",
                    rc_f_train,
                    rc_f_test,
                ]
            )

        # Remove saved representations to save space
        for module_name in module_dic:
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            if os.path.exists(representation_dir/f'{hook_name}.npy'):
                os.remove(representation_dir/f'{hook_name}.npy')
            hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
            if os.path.exists(representation_dir/f'{hook_name}.npy'):
                os.remove(representation_dir/f'{hook_name}.npy')

    results_df = pd.DataFrame(
        results_data,
        columns=["Concept", "Layer", "Seed", "Method", "Train ACC", "Test ACC"],
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_concept_accuracy(save_dir, None, "mnist")
        for concept in concept_to_class:
            plot_concept_accuracy(save_dir, concept, "mnist")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 2)))
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=60)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--concept_accuracy", action="store_true")
    parser.add_argument("--concept_disentanglement", action="store_true")

    args = parser.parse_args()
    model_name = f"model_{args.latent_dim}"
    if args.train:
        train_mnist_model(args.latent_dim, args.batch_size, model_name=model_name)
    if args.concept_accuracy:
        concept_accuracy(args.seeds, args.latent_dim, args.plot, model_name=model_name)
    if args.concept_disentanglement:
        concept_disentanglement([0], args.latent_dim, args.plot, model_name=model_name)