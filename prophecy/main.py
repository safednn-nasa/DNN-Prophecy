import argparse
import pandas as pd

from prophecy.data.dataset import Dataset
from prophecy.utils.misc import get_model, lookup_settings, load_settings
from prophecy.core.extract import RuleExtractor
from prophecy.core.detect import Detector
from prophecy.utils.paths import results_path

SETTINGS = lookup_settings()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='target dataset', required=True, choices=['PD'])
    parser.add_argument('--settings', type=str, help='settings', required=True, choices=SETTINGS.keys())

    subparsers = parser.add_subparsers(dest='subparser')
    detect_parser = subparsers.add_parser('detect')
    detect_parser.add_argument('-t', '--threshold', type=float, help='rule F1-threshold', default=0.0)

    extract_parser = subparsers.add_parser('extract')

    args = parser.parse_args()
    model = get_model(args.model, args.version)
    dataset = Dataset(args.dataset)
    settings = load_settings(SETTINGS[args.settings])

    base_path = results_path / f"{args.model}{args.version}"
    rules_path = base_path / "rules" / settings.rules / settings.fingerprint
    rules_path.mkdir(parents=True, exist_ok=True)
    predictions_path = base_path / "predictions" / settings.rules / settings.fingerprint
    predictions_path.mkdir(parents=True, exist_ok=True)

    if args.subparser == 'extract':
        rule_extractor = RuleExtractor(model=model, dataset=dataset, settings=settings)
        ruleset = rule_extractor()

        for layer, rules in ruleset.items():
            df = pd.DataFrame(rules)

            if len(df) == 0:
                continue

            df.to_csv(rules_path / f'{layer}.csv', index=False)

    elif args.subparser == 'detect':
        detector = Detector(model=model, dataset=dataset)

        dfs = []
        for f in rules_path.iterdir():
            if f.is_file() and f.suffix == '.csv':
                layer = f.stem
                rules = pd.read_csv(f)
                rules['layer'] = layer
                dfs.append(rules)

        ruleset = pd.concat(dfs)
        ruleset = ruleset[ruleset['f1'] >= args.threshold]
        results = detector(ruleset)
        pd.DataFrame([results]).to_csv(predictions_path / 'results.csv', index=False)
    else:
        print("Please specify a command ['extract', 'detect'].")
        exit()
