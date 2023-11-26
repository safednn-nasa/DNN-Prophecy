import argparse
import pandas as pd

from prophecy.data.dataset import Dataset
from prophecy.utils.misc import get_model, lookup_settings, load_settings
from prophecy.core.extract import RuleExtractor
from prophecy.utils.paths import results_path

SETTINGS = lookup_settings()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='target dataset', required=True, choices=['PD'])
    parser.add_argument('--settings', type=str, help='settings', required=True, choices=SETTINGS.keys())
    # parser.add_argument('--split', type=str, help='Dataset split', required=True,
    #                    choices=['train', 'val', 'unseen'])
    args = parser.parse_args()
    
    model = get_model(args.model, args.version)
    dataset = Dataset(args.dataset)
    settings = load_settings(SETTINGS[args.settings])
    rule_extractor = RuleExtractor(model=model, dataset=dataset, settings=settings)
    ruleset = rule_extractor()

    output_path = results_path / f"{args.model}{args.version}" / "rules" / settings.rules / settings.fingerprint
    output_path.mkdir(parents=True, exist_ok=True)

    for layer, rules in ruleset.items():
        df = pd.DataFrame(rules)

        if len(df) == 0:
            continue

        df.to_csv(output_path / f'{layer}.csv', index=False)
