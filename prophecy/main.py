import sys
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from prophecy.utils.misc import get_model, read_split
from prophecy.core.extract import Extractor
from prophecy.core.detect import RulesDetector, ClassifierDetector
from prophecy.utils.paths import results_path


def run_analyze_command():
    train_features, train_labels = read_split(args.train_features, args.train_labels)
    val_features, val_labels = read_split(args.val_features, args.val_labels)

    rule_extractor = Extractor(model=model, train_features=train_features, train_labels=train_labels,
                               val_features=val_features, val_labels=val_labels, skip_rules=args.skip_rules,
                               only_dense=args.only_dense_layers, balance=args.balance, confidence=args.confidence,
                               only_activation=args.only_activation_layers)

    ruleset = rule_extractor(path=classifiers_path)
    pd.DataFrame(ruleset).to_csv(rules_path, index=False)


def run_classify_command():
    test_features, test_labels = read_split(args.test_features, args.test_labels)

    file_name = 'results_clf_pure.csv' if args.only_pure else 'results_clf.csv'
    output_path = predictions_path / file_name
    clf_detector = ClassifierDetector(model=model, learners_path=classifiers_path, features=test_features,
                                      labels=test_labels, only_pure=args.only_pure)
    results = clf_detector()

    pd.DataFrame(results).to_csv(output_path, index=False)
    pd.DataFrame(clf_detector.stats).to_csv(predictions_path / 'stats.csv', index=False)


def run_detect_command():
    test_features, test_labels = read_split(args.test_features, args.test_labels)

    output_path = predictions_path / 'results.csv'
    ruleset = pd.read_csv(rules_path)
    ruleset = ruleset[ruleset['f1'] >= args.threshold]

    detector = RulesDetector(model=model, ruleset=ruleset, features=test_features, labels=test_labels)
    results = detector()

    pd.DataFrame(results).to_csv(output_path, index=False)
    pd.DataFrame(detector.stats).to_csv(predictions_path / 'stats.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('-m', '--model_path', type=str, help='Model to infer the precondition',
                        required=True)
    parser.add_argument('-wd', '--workdir', type=str, help='Working directory', required=False)

    action_parser = parser.add_subparsers(dest='action')
    infer_parser = action_parser.add_parser('infer')
    infer_parser.add_argument('-tx', '--test_features', type=str, help='Test features', required=True)
    infer_parser.add_argument('-ty', '--test_labels', type=str, help='Test labels', required=True)

    infer_subparser = infer_parser.add_subparsers(dest='infer_subparser')
    rules_parser = infer_subparser.add_parser('rules')
    rules_parser.add_argument('-t', '--threshold', type=float, help='rule F1-threshold', default=0.0)

    classifiers_parser = infer_subparser.add_parser('classifiers')
    classifiers_parser.add_argument('-op', '--only-pure', action='store_true', default=False,
                                    help='Consider only classifications with 100 probability')

    analyze_parser = action_parser.add_parser('analyze')
    analyze_parser.add_argument('-tx', '--train_features', type=str, help='Train features', required=True)
    analyze_parser.add_argument('-ty', '--train_labels', type=str, help='Train labels', required=True)
    analyze_parser.add_argument('-vx', '--val_features', type=str, help='Validation features', required=True)
    analyze_parser.add_argument('-vy', '--val_labels', type=str, help='Validation labels', required=True)
    analyze_parser.add_argument('-odl', '--only-dense-layers', action='store_true', default=False,
                                help='Consider only dense layers')
    analyze_parser.add_argument('-oal', '--only-activation-layers', action='store_true', default=False,
                                help='Include the activation layers associated to the dense layers')
    analyze_parser.add_argument('-sr', '--skip-rules', action='store_true', default=False,
                                help='Skip rules extraction')
    analyze_parser.add_argument('-b', '--balance', action='store_true', default=False,
                                help='Balance classes in the dataset for training the classifiers.')
    analyze_parser.add_argument('-c', '--confidence', action='store_true', default=False,
                                help='Adjust labels in the dataset for training the classifiers with the confidence.')
    analyze_parser.add_argument('-rs', '--random-state', type=int, help='Random state for reproducibility',
                                default=42)

    args = parser.parse_args()
    model = get_model(args.model_path)

    working_dir = Path(args.workdir) if args.workdir else results_path

    rules_path = working_dir / 'ruleset.csv'
    classifiers_path = working_dir / "classifiers"
    classifiers_path.mkdir(parents=True, exist_ok=True)
    predictions_path = working_dir / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)

    if args.action == 'analyze':
        run_analyze_command()
    elif args.action == 'infer':
        if args.infer_subparser == 'rules':
            run_detect_command()
        elif args.infer_subparser == 'classifiers':
            run_classify_command()
        else:
            print("Please specify a command ['rules', 'classifiers'].", file=sys.stderr)
    else:
        print("Please specify a command ['analyze', 'infer'].", file=sys.stderr)
        exit()
