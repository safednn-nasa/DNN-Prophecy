import argparse
import pandas as pd
import numpy as np

from prophecy.utils.misc import lookup_models, get_model
from prophecy.core.extract import Extractor
from prophecy.core.detect import RulesDetector, ClassifierDetector
from prophecy.utils.paths import results_path

MODELS = lookup_models()
model_choices = sorted(list(MODELS.keys()))


def run_extract_command():
    train_features = pd.read_csv(args.train_features)
    train_labels = np.loadtxt(args.train_labels, dtype=int)
    val_features = pd.read_csv(args.val_features)
    val_labels = np.loadtxt(args.val_labels, dtype=int)

    rule_extractor = Extractor(model=model, train_features=train_features, train_labels=train_labels,
                               val_features=val_features, val_labels=val_labels, skip_rules=args.skip_rules,
                               only_dense=args.only_dense_layers, balance=args.balance, confidence=args.confidence,
                               include_activation=args.include_activation_layers)

    ruleset = rule_extractor(path=classifiers_path)

    # TODO: maybe move this in the rule extractor class
    for layer, rules in ruleset.items():
        df = pd.DataFrame(rules)

        if len(df) == 0:
            continue

        df.to_csv(rules_path / f'{layer}.csv', index=False)


def run_classify_command():
    test_features = pd.read_csv(args.test_features)
    test_labels = np.loadtxt(args.test_labels, dtype=int)

    file_name = 'results_clf_pure.csv' if args.only_pure else 'results_clf.csv'
    output_path = predictions_path / file_name
    clf_detector = ClassifierDetector(model=model, learners_path=classifiers_path, features=test_features,
                                      labels=test_labels, only_pure=args.only_pure)
    results = clf_detector()
    pd.DataFrame(results, index=[0]).to_csv(output_path, index=False)


def run_detect_command():
    test_features = pd.read_csv(args.test_features)
    test_labels = np.loadtxt(args.test_labels, dtype=int)

    results = []
    output_path = predictions_path / 'results.csv'

    dfs = []

    for f in rules_path.iterdir():
        if f.is_file() and f.suffix == '.csv':
            layer = f.stem
            rules = pd.read_csv(f)
            rules['layer_count'] = rules['layer']
            rules['layer'] = layer
            dfs.append(rules)

    ruleset = pd.concat(dfs)
    ruleset = ruleset[ruleset['f1'] >= args.threshold]

    detector = RulesDetector(model=model, ruleset=ruleset, features=test_features, labels=test_labels)
    results.append(detector())

    pd.DataFrame(results).to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=model_choices)

    subparsers = parser.add_subparsers(dest='subparser')
    detect_parser = subparsers.add_parser('detect')
    detect_parser.add_argument('-tx', '--test_features', type=str, help='Test features', required=True)
    detect_parser.add_argument('-ty', '--test_labels', type=str, help='Test labels', required=True)
    detect_parser.add_argument('-t', '--threshold', type=float, help='rule F1-threshold', default=0.0)

    classify_parser = subparsers.add_parser('classify')
    classify_parser.add_argument('-tx', '--test_features', type=str, help='Test features', required=True)
    classify_parser.add_argument('-ty', '--test_labels', type=str, help='Test labels', required=True)
    classify_parser.add_argument('-op', '--only-pure', action='store_true', default=False,
                                 help='Consider only classifications with 100 probability')

    extract_parser = subparsers.add_parser('extract')
    extract_parser.add_argument('-tx', '--train_features', type=str, help='Train features', required=True)
    extract_parser.add_argument('-ty', '--train_labels', type=str, help='Train labels', required=True)
    extract_parser.add_argument('-vx', '--val_features', type=str, help='Validation features', required=True)
    extract_parser.add_argument('-vy', '--val_labels', type=str, help='Validation labels', required=True)
    extract_parser.add_argument('-odl', '--only-dense-layers', action='store_true', default=False,
                                help='Consider only dense layers')
    extract_parser.add_argument('-ial', '--include-activation-layers', action='store_true', default=False,
                                help='Include the activation layers associated to the dense layers')
    extract_parser.add_argument('-sr', '--skip-rules', action='store_true', default=False,
                                help='Skip rules extraction')
    extract_parser.add_argument('-b', '--balance', action='store_true', default=False,
                                help='Balance classes in the dataset for training the classifiers.')
    extract_parser.add_argument('-c', '--confidence', action='store_true', default=False,
                                help='Adjust labels in the dataset for training the classifiers with the confidence.')

    args = parser.parse_args()

    model = get_model(MODELS[args.model])
    base_path = results_path / args.model
    rules_path = base_path / "rules"
    rules_path.mkdir(parents=True, exist_ok=True)
    classifiers_path = base_path / "classifiers"
    classifiers_path.mkdir(parents=True, exist_ok=True)
    predictions_path = base_path / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)

    if args.subparser == 'extract':
        run_extract_command()
    elif args.subparser == 'detect':
        run_detect_command()
    elif args.subparser == 'classify':
        run_classify_command()
    else:
        print("Please specify a command ['extract', 'detect'].")
        exit()
