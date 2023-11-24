import argparse

from prophecy.data.dataset import Dataset
from prophecy.utils.misc import get_model
from prophecy.core.extract import extract_rules


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='target dataset', required=True, choices=['PD'])
    parser.add_argument('--split', type=str, help='Dataset split', required=True,
                        choices=['train', 'val', 'unseen'])
    args = parser.parse_args()
    
    model = get_model(args.model, args.version)
    dataset = Dataset(args.dataset)

    extract_rules(model, dataset, args.split)
