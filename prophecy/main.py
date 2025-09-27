#* * * * * * * * * * * * * * 
#Notices:
#
#Copyright © 2025 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  
#All Rights Reserved.
#
#Disclaimers
#
#No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, 
#INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, 
#OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE 
#AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER 
#APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING 
#THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
#
#Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS,
#AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES 
#ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT 
#SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE 
#EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT. 
#* * * * * * * * * * * * * * 
import sys
import os
import argparse
import numpy as np
import pandas as pd
import shutil

from pathlib import Path

from prophecy.utils.misc import get_model, read_split
from prophecy.core.extract import Extractor
from prophecy.core.detect import RulesDetector, ClassifierDetector
from prophecy.core.proof import RulesProve
from prophecy.utils.paths import results_path


def run_analyze_command():
    train_features, train_labels = read_split(args.train_features, args.train_labels)
    val_features, val_labels = read_split(args.val_features, args.val_labels)

    rule_extractor = Extractor(model=model, train_features=train_features, train_labels=train_labels,
                               val_features=val_features, val_labels=val_labels, skip_rules=args.skip_rules, layer_name=args.layer_name,
                               only_dense=args.only_dense_layers, balance=args.balance, confidence=args.confidence,
                               only_activation=args.only_activation_layers, type=args.type, inptype=args.inptype, acts=args.acts, top=args.top)
    
    
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

def run_prove_command(lab: int):
    train_features, train_labels = read_split(args.train_features, args.train_labels)
    val_features = []
    val_labels = []
    if (args.val_features != None and args.val_labels != None):
        val_features, val_labels = read_split(args.val_features, args.val_labels)

    output_path = predictions_path / 'results.txt'

    ruleset = pd.read_csv(rules_path)
    ruleset = ruleset[ruleset['label'] == lab]
    ruleset = ruleset.sort_values(by=['support'], ascending=False)
    ruleset = ruleset.reset_index()
    ruleset = ruleset[ruleset.index == 0]

    
    print("PROVE RULE for Label:", lab)
    print("RULE WITH HIGHEST SUPPORT ON TRAIN DATA")
    print("LAYER, NEURONS AND SIGNATURE:")
    top_rule_layer = ruleset['layer']
    top_rule_layer_nm = (top_rule_layer.array[0]).strip()
    print("LAYER:", top_rule_layer_nm)

    rule_neurons_df = ruleset['neurons']
    rule_neurons_list = []
    rule_neurons = (rule_neurons_df.array[0]).split(",")
    for indx in range(0, len(rule_neurons)):
        rule_neurons[indx] = (rule_neurons[indx]).strip()
        rule_neurons[indx] = (rule_neurons[indx]).replace("[", "")
        rule_neurons[indx] = (rule_neurons[indx]).replace("]","")
        rule_neurons_list.append(int(rule_neurons[indx]))
    
    print("NEURONS:",rule_neurons_list)

    rule_sig_df = ruleset['signature']
    rule_sig_list = []
    rule_sig = (rule_sig_df.array[0]).split(",")
    for indx in range(0, len(rule_sig)):
        rule_sig[indx] = (rule_sig[indx]).strip()
        rule_sig[indx] = (rule_sig[indx]).replace("[", "")
        rule_sig[indx] = (rule_sig[indx]).replace("]","")
        if (len(rule_sig) == len(rule_neurons)):
            rule_sig_list.append(int(rule_sig[indx]))
        else:
            if (indx % 2 == 0):
                rule_sig[indx] = (rule_sig[indx]).replace("'", "")
                rule_sig_list.append(rule_sig[indx])
            else:
                rule_sig_list.append(float(rule_sig[indx]))
        
    print("SIGNATURE:",rule_sig_list)

    print("ONNX MODEL:", onnx_model)
    print("ONNX MAP:", onnx_map)
    print("FEATURES:", np.shape(train_features))
    print("LABELS:", np.shape(train_labels))

    #source_file1 = '/content/ProphecyPlus/dataset_models/MarabouNetworkONNX.py'
    #source_file2 = '/content/ProphecyPlus/dataset_models/ONNXParser.py'
    
    #destination_file1 = marabou_path + '/MarabouNetworkONNX.py'
    #destination_file2 = marabou_path + '/parsers/ONNXParser.py'

    #shutil.copy(source_file1, destination_file1)
    #shutil.copy(source_file2, destination_file2)

    print("PATH:", marabou_path)

    results = False
    it = 0
    if (pred_post == False):
        print("CONSTRAINTS PATH:", consts_path)
        #conditions = np.genfromtxt(consts_path, delimiter=',', dtype=str)
        #print(np.shape(conditions))
        unsolved_labs = []
        while (results == False):
            print("ITERATION #:", it)
            prove_marabou = RulesProve(model=model, onnx_model_nm=onnx_model, onnx_map_nm=onnx_map, layer_nm = top_rule_layer_nm, neurons=rule_neurons_list, sig=rule_sig_list,features=train_features, labels=train_labels,lab=lab,iter=it,unsolved = unsolved_labs, min_const=min_const,pred_post=pred_post, op_consts=consts_path, Vfeatures=val_features, Vlabels=val_labels)
            results,unsolved = prove_marabou()
            it = it + 1
        
    if (pred_post == True):
        unsolved_labs = []
        while (results == False):
            print("ITERATION #:", it)
            print("UNSOLVED LABELS:", unsolved_labs)
            prove_marabou = RulesProve(model=model, onnx_model_nm=onnx_model, onnx_map_nm=onnx_map, layer_nm = top_rule_layer_nm, neurons=rule_neurons_list, sig=rule_sig_list,features=train_features, labels=train_labels,lab=lab,iter=it,unsolved = unsolved_labs, min_const=min_const,pred_post=pred_post, op_consts=None, Vfeatures=val_features, Vlabels=val_labels)
            results,unsolved = prove_marabou()
            unsolved_labs = []
            for indx in range(0,len(unsolved)):
                unsolved_labs.append(unsolved[indx])
            if (it == -1):
                break
            it = it + 1
   
        


if __name__ == '__main__':
    path = os.environ['PATH']
    print(path)
    #os.environ['PATH'] = path + ':../Marabou/:../Marabou/build:../Marabou/build/bin'
    #print(os.environ['PATH'])
    parser = argparse.ArgumentParser(description='Prophecy')
    parser.add_argument('-m', '--model_path', type=str, help='Model in keras (.h5) format.',
                        required=False)
    parser.add_argument('-wd', '--workdir', type=str, help='Working directory', required=False)

    action_parser = parser.add_subparsers(dest='action')
    
    monitor_parser = action_parser.add_parser('monitor')
    monitor_parser.add_argument('-tx', '--test_features', type=str, help='Test features', required=True)
    monitor_parser.add_argument('-ty', '--test_labels', type=str, help='Test labels', required=True)

    monitor_subparser = monitor_parser.add_subparsers(dest='monitor_subparser')
    rules_parser = monitor_subparser.add_parser('rules')
    rules_parser.add_argument('-t', '--threshold', type=float, help='rule F1-threshold', default=0.0)

    classifiers_parser = monitor_subparser.add_parser('classifiers')
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
    analyze_parser.add_argument('-type', '--type', type=int, help='Predictions based rules: 0, Accuracy rules: 1, Accuracy per label rules: 2, Labels_array based: 3',
                                default=1)
    analyze_parser.add_argument('-inptype', '--inptype', type=int, help='Model: 0, Neuron_acts_array: 1',
                                default=0)
    analyze_parser.add_argument('-acts', '--acts', type=bool, help='On/Off: True, Values: False',
                                default=False)
    analyze_parser.add_argument('-layer_name', '--layer_name', type=str, help='Name of dense or activation layer',
                                default=None)
    analyze_parser.add_argument('-top', '--top', type=bool, help='Top rules or All rules',
                                default=False)

    prove_parser = action_parser.add_parser('prove')
    prove_parser.add_argument('-mp', '--marabou_path', type=str, help='path to Marabou folder', required=True)
    prove_parser.add_argument('-onx', '--onnx_path', type=str, help='model in ONNX form', required=True)
    prove_parser.add_argument('-onx_map', '--onnx_map', type=str, help='map between the layers of .h5 and .onnx models', required=True)
    prove_parser.add_argument('-tx', '--train_features', type=str, help='Train features', required=True)
    prove_parser.add_argument('-ty', '--train_labels', type=str, help='Train labels', required=False)
    prove_parser.add_argument('-vx', '--val_features', type=str, help='Val features', required=False)
    prove_parser.add_argument('-vy', '--val_labels', type=str, help='Val labels', required=False)
    prove_parser.add_argument('-label', '--lab', type=int, default=-1,
                                help='select top rules for given label.')
    prove_parser.add_argument('-min_const', '--min_const', type=bool, help='output constraints', default=False)
    prove_parser.add_argument('-pred', '--pred', type=bool, help='prediction post', default=False)
    prove_parser.add_argument('-cp', '--cp', type=str, help='path to output constraints file', default=None)
    

    args = parser.parse_args()

    if ((args.action == 'analyze') and (args.inptype == 1)):
        model = None
    else:
        model = get_model(args.model_path)

    if (args.action == 'prove'):
        onnx_model = args.onnx_path
        onnx_map = args.onnx_map
        marabou_path = args.marabou_path
        min_const = args.min_const
        pred_post = args.pred
        consts_path = args.cp
        
    working_dir = Path(args.workdir) if args.workdir else results_path

    rules_path = working_dir / 'ruleset.csv'
    classifiers_path = working_dir / "classifiers"
    classifiers_path.mkdir(parents=True, exist_ok=True)
    predictions_path = working_dir / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)

    if args.action == 'analyze':
        run_analyze_command()
    elif args.action == 'prove':
        run_prove_command(args.lab)
    elif args.action == 'monitor':
        if args.monitor_subparser == 'rules':
            run_detect_command()
        elif args.monitor_subparser == 'classifiers':
            run_classify_command()
        else:
            print("Please specify a command ['rules', 'classifiers'].", file=sys.stderr)
    else:
        print("Please specify a command ['analyze', 'monitor', 'prove'].", file=sys.stderr)
        exit()
