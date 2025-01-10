import torch
import os
import logging
import argparse
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from feature_mining.decision_tree_concept import DTExplainer
from feature_mining.entanglement import *
from tqdm import tqdm
from utils.plot import (
    plot_concept_disentanglement,
    plot_concept_accuracy,
    plot_global_explanation,
    plot_attribution_correlation,
    plot_color_saliency,
)
from sklearn.metrics import accuracy_score
from pathlib import Path
from utils.dataset import load_cub_data, CUBDataset, generate_cub_concept_dataset
from models.cub import CUBClassifier, CUBResNet
from utils.hooks import register_hooks, remove_all_hooks, get_saved_representations

train_path = str(Path.cwd() / "data/cub/CUB_processed/class_attr_data_10/train.pkl")
val_path = str(Path.cwd() / "data/cub/CUB_processed/class_attr_data_10/val.pkl")
test_path = str(Path.cwd() / "data/cub/CUB_processed/class_attr_data_10/test.pkl")
img_dir = str(Path.cwd() / f"data/cub/CUB_200_2011")
img_adv_dir = str(Path.cwd() / f"data/cub/CUB_random")


def fit_model(batch_size: int, n_epochs: int, model_name: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_dir = Path.cwd() / f"results/cub/{model_name}"
    if not model_dir.exists():
        os.makedirs(model_dir)
    train_loader = load_cub_data(
        [train_path, val_path],
        use_attr=False,
        batch_size=batch_size,
        image_dir=img_dir,
        no_img=False,
    )
    test_loader = load_cub_data(
        [test_path],
        use_attr=False,
        batch_size=batch_size,
        image_dir=img_dir,
        no_img=False,
    )
    if model_name == "inception_model":
        model = CUBClassifier(name=model_name)
    else:
        model = CUBResNet(name=model_name)
    # model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.fit(
        device, train_loader, test_loader, model_dir, patience=50, n_epoch=n_epochs
    )


def concept_disentanglement(
    random_seeds: list[int],
    plot: bool,
    batch_size: int,
    save_dir: Path = Path.cwd() / "results/cub/concept_disentanglement",
    model_dir: Path = Path.cwd() / f"results/cub/",
    model_name: str = "model",
    num_concept: int = 112
):  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])
    # save representation to local runtime to save drive space
    representation_dir = Path("/root") / f"{model_name}_representations"
    # representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)
    # Load model
    model_dir = model_dir / model_name
    if model_name == "inception_model":
        model = CUBClassifier(name=model_name)
    else:
        model = CUBResNet(name=model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    concept_names = train_set.get_concept_names()
    concept_names.insert(0, "Nonesense")
    concept_names = concept_names[:num_concept]

    module_names = []
    dise_train = []
    dise_test = []
    for concept_id, concept_name in enumerate(concept_names):
        concept_id -= 1
        for random_seed in random_seeds:
            logging.info(f"Working with concept {concept_name} and seed {random_seed}")
            # Save representations for training concept examples and then remove the hooks
            module_dic, handler_train_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_train"
            )
            X_train, y_train = generate_cub_concept_dataset(
                concept_id,
                350,
                random_seed,
                [train_path, val_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_train in np.array_split(X_train, batch_size):
                model(torch.from_numpy(x_train).to(device))
            remove_all_hooks(handler_train_dic)
            # Save representations for testing concept examples and then remove the hooks
            module_dic, handler_test_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_test"
            )
            X_test, y_test = generate_cub_concept_dataset(
                concept_id,
                100,
                random_seed,
                [test_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_test in np.array_split(X_test, batch_size):
                model(torch.from_numpy(x_test).to(device))
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
        index=concept_names,
    ).T
    dise_test_df = pd.DataFrame(
        dise_test,
        columns=module_names,
        index=concept_names,
    ).T

    csv_train_path = save_dir / "metrics_train.csv"
    csv_test_path = save_dir / "metrics_test.csv"
    dise_train_df.to_csv(csv_train_path, header=True, mode="w", index=True)
    dise_test_df.to_csv(csv_test_path, header=True, mode="w", index=True)
    
    if plot:
        plot_concept_disentanglement(save_dir, "cub", train=True, fig_size=18)
        plot_concept_disentanglement(save_dir, "cub", train=False, fig_size=18)


def concept_accuracy(
    random_seeds: list[int],
    plot: bool,
    batch_size: int,
    save_dir: Path = Path.cwd() / "results/cub/concept_accuracy",
    model_dir: Path = Path.cwd() / f"results/cub/",
    model_name: str = "model",
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(random_seeds[0])
    # save representation to local runtime to save drive space
    representation_dir = Path("/root") / f"{model_name}_representations"
    # representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    # Load model
    model_dir = model_dir / model_name
    if model_name == "inception_model":
        model = CUBClassifier(name=model_name)
    else:
        model = CUBResNet(name=model_name)
    model.load_state_dict(torch.load(model_dir / f"{model_name}.pt"), strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    train_set = CUBDataset(
        [train_path, val_path],
        use_attr=True,
        no_img=False,
        uncertain_label=False,
        image_dir=img_dir,
        n_class_attr=2,
    )
    concept_names = train_set.get_concept_names()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_id, concept_name in enumerate(concept_names):
        for random_seed in random_seeds:
            logging.info(f"Working with concept {concept_name} and seed {random_seed}")
            # Save representations for training concept examples and then remove the hooks
            module_dic, handler_train_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_train"
            )
            X_train, y_train = generate_cub_concept_dataset(
                concept_id,
                400,
                random_seed,
                [train_path, val_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_train in np.array_split(X_train, batch_size):
                model(torch.from_numpy(x_train).to(device))
            remove_all_hooks(handler_train_dic)
            # Save representations for testing concept examples and then remove the hooks
            module_dic, handler_test_dic = register_hooks(
                model, representation_dir, f"{concept_name}_seed{random_seed}_test"
            )
            X_test, y_test = generate_cub_concept_dataset(
                concept_id,
                100,
                random_seed,
                [test_path],
                False,
                False,
                image_dir=img_dir,
            )
            for x_test in np.array_split(X_test, batch_size):
                model(torch.from_numpy(x_test).to(device))
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
                acc_t_train, rc_t_train, acc_f_train, rc_f_train = dt.validate(H_train, y_train, 15)
                acc_t_test, rc_t_test, acc_f_test, rc_f_test = dt.validate(H_test, y_test, 15)
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
        plot_concept_accuracy(save_dir, None, "cub")

    results_md = pd.pivot_table(
        results_df,
        index="Layer",
        columns="Method",
        aggfunc=["mean", "sem"],
        values="Test ACC",
    ).to_markdown()
    logging.info(results_md)



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--concept_category", type=str, default="Primary Color")
    parser.add_argument("--samples_per_concept", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="inception")
    parser.add_argument("--car_sensitivity", action="store_true")
    parser.add_argument("--concept_accuracy", action="store_true")
    parser.add_argument("--concept_disentanglement", action="store_true")
    parser.add_argument("--num_concept", type=int, default=112)
    args = parser.parse_args()

    assert args.model_type in {"resnet", "inception"}
    model_name = f"{args.model_type}_model"
    if args.train:
        fit_model(args.batch_size, args.n_epochs, model_name=model_name)
    if args.concept_accuracy:
        concept_accuracy(args.seeds, args.plot, args.batch_size, model_name=model_name)
    if args.concept_disentanglement:
        concept_disentanglement([0], args.plot, args.batch_size, model_name=model_name, num_concept=args.num_concept)