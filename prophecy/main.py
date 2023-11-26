import argparse

from prophecy.data.dataset import Dataset
from prophecy.utils.misc import get_model, lookup_settings, load_settings
from prophecy.core.extract import RuleExtractor

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
    rule_extractor = RuleExtractor(model, dataset, settings=settings)
    rule_extractor()
