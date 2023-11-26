import pandas as pd
import numpy as np
import operator

from tqdm import tqdm


def get_all_invariants(estimator):
    # Returns a dictionary mapping each decision tree prediction class
    # to a list of invariants. Each invariant is specified as a triple:
    # - neuron ids
    # - neuron signature (for the neuron ids)
    # - number of training samples that hit it
    # The neuron ids and neuron signature can be supplied to get_suffix_cluster
    # to obtain the cluster of training instances that hit the invariant.
    def is_leaf(node):
        return estimator.tree_.children_left[node] == estimator.tree_.children_right[node]

    def left_child(node):
        return estimator.tree_.children_left[node]

    def right_child(node):
        return estimator.tree_.children_right[node]

    def get_all_paths_rec(node):
        # Returns a list of triples corresponding to paths
        # in the decision tree. Each triple consists of
        # - neurons encountered along the path
        # - signature along the path
        # - prediction class at the leaf
        # - number of training samples that hit the path
        # The prediction class and number of training samples
        # are set to -1 when the leaf is "impure".
        feature = estimator.tree_.feature
        if is_leaf(node):
            values = estimator.tree_.value[node][0]
            if len(np.where(values != 0)[0]) == 1:
                cl = estimator.classes_[np.where(values != 0)[0][0]]
                nsamples = estimator.tree_.n_node_samples[node]
            else:
                # impure node
                cl = -1
                nsamples = -1
            return [[[], [], cl, nsamples]]
        # If it is not a leaf both left and right childs must exist
        paths = [[[feature[node]] + p[0], [0] + p[1], p[2], p[3]] for p in get_all_paths_rec(left_child(node))]
        paths += [[[feature[node]] + p[0], [1] + p[1], p[2], p[3]] for p in get_all_paths_rec(right_child(node))]
        return paths

    paths = get_all_paths_rec(0)
    print("Obtained all paths")
    invariants = {}
    for p in tqdm(paths):
        neuron_ids, neuron_sig, cl, nsamples = p
        if cl not in invariants:
            invariants[cl] = []
        # cluster = get_suffix_cluster(neuron_ids, neuron_sig)
        invariants[cl].append([neuron_ids, neuron_sig, nsamples])
    for cl in invariants.keys():
        invariants[cl] = sorted(invariants[cl], key=operator.itemgetter(2), reverse=True)
    return invariants


def get_all_invariants_val(estimator):
    # Returns a dictionary mapping each decision tree prediction class
    # to a list of invariants. Each invariant is specified as a triple:
    # - neuron ids
    # - neuron signature (for the neuron ids)
    # - number of training samples that hit it
    # The neuron ids and neuron signature can be supplied to get_suffix_cluster
    # to obtain the cluster of training instances that hit the invariant.
    def is_leaf(node):
        return estimator.tree_.children_left[node] == estimator.tree_.children_right[node]

    def left_child(node):
        return estimator.tree_.children_left[node]

    def right_child(node):
        return estimator.tree_.children_right[node]

    def get_all_paths_rec(node):
        # Returns a list of triples corresponding to paths
        # in the decision tree. Each triple consists of
        # - neurons encountered along the path
        # - signature along the path
        # - prediction class at the leaf
        # - number of training samples that hit the path
        # The prediction class and number of training samples
        # are set to -1 when the leaf is "impure".
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        if is_leaf(node):
            values = estimator.tree_.value[node][0]
            if len(np.where(values != 0)[0]) == 1:
                cl = estimator.classes_[np.where(values != 0)[0][0]]
                nsamples = estimator.tree_.n_node_samples[node]
            else:
                # impure node
                cl = -1
                nsamples = -1
            return [[[], [], cl, nsamples]]
        # If it is not a leaf both left and right childs must exist
        # paths = [[[feature[node]] + p[0], [0] + p[1], p[2], p[3]] for p in get_all_paths_rec(left_child(node))]
        # paths += [[[feature[node]] + p[0], [1] + p[1], p[2], p[3]] for p in get_all_paths_rec(right_child(node))]
        paths = [[[feature[node]] + p[0], ['<='] + [threshold[node]] + p[1], p[2], p[3]] for p in
                 get_all_paths_rec(left_child(node))]
        paths += [[[feature[node]] + p[0], ['>'] + [threshold[node]] + p[1], p[2], p[3]] for p in
                  get_all_paths_rec(right_child(node))]
        return paths

    paths = get_all_paths_rec(0)
    print("Obtained all paths")
    invariants = {}
    for p in tqdm(paths):
        neuron_ids, neuron_sig, cl, nsamples = p
        if cl not in invariants:
            invariants[cl] = []
        # cluster = get_suffix_cluster(neuron_ids, neuron_sig)
        invariants[cl].append([neuron_ids, neuron_sig, nsamples])
    for cl in invariants.keys():
        invariants[cl] = sorted(invariants[cl], key=operator.itemgetter(2), reverse=True)
    return invariants


# print("MIS-CLASSIFIED:", total_fail)

def check_pattern(layer_vals, neuron_ids, neuron_sig):
    found = True
    oper = -1
    # layer_vals = (layer_vals).flatten()

    for ind in range(0, len(neuron_sig)):
        if (ind % 2 == 0):
            op = neuron_sig[ind]
            if (op == '<='):
                oper = 0
            else:
                oper = 1
        else:
            v = int(neuron_ids[(int)(ind / 2)])
            vsig = float(neuron_sig[ind])
            val = float(layer_vals[v])
            # print(v,vsig,val,oper)
            if (oper == 0):
                if (val > vsig):
                    # print(v,val,vsig,oper)
                    found = False
                    break
            else:
                if (val <= vsig):
                    # print(v,val,vsig,oper)
                    found = False
                    break
            oper = -1

    return found


def get_suffix_cluster(neuron_ids, neuron_sig, suffixes, VAL=False):
    # Get the cluster of inputs that such that all inputs in the cluster
    # have provided on/off signature for the provided neurons.
    #
    # The returned cluster is an array of indices (into mnist.train.images).
    if (VAL == False):
        return np.where((suffixes[:, neuron_ids] == neuron_sig).all(axis=1))[0]

    matched_ids = []
    # print(len(suffixes))
    for indx in range(0, len(suffixes)):
        if (check_pattern(suffixes[indx], neuron_ids, neuron_sig) == True):
            matched_ids.append(indx)
    # print(matched_ids)
    return matched_ids


def calc_prec_recall(suffixes, labels, neurons, signature, cl, VAL, supp=-1):
    if (cl not in set(labels)):
        return (-1, -1, -1)

    TOT_LABELS = len(set(labels))
    total_labels = np.zeros(TOT_LABELS - 1)
    # print("TOTAL LABELS:",TOT_LABELS)
    for indx in range(0, TOT_LABELS - 1):
        total_labels[indx] = len(np.where(labels == indx)[0])

    total_fail = len(np.where(labels == 1000)[0])

    recall = 0
    prec = 0

    if (supp != -1):
        prec = 100
        if (cl != 1000):
            recall = (supp / (total_labels[cl])) * 100.0
        else:
            recall = (supp / (total_fail)) * 100.0
        coverage = (supp / len(suffixes)) * 100.0
        return (coverage, prec, recall)

    cls = get_suffix_cluster(neurons, signature, suffixes, VAL)
    cls_labels = []
    for indx1 in range(0, len(cls)):
        cls_labels.append(labels[cls[indx1]])
    coverage = (len(cls) / len(suffixes)) * 100.0

    if (cl != 1000):
        true_pos = len(np.where(cls_labels == cl)[0])
        false_pos = len(cls) - true_pos
        false_neg = (total_labels[cl]) - true_pos
        if ((true_pos + false_neg) == 0):
            recall = 0
        else:
            recall = (true_pos / (true_pos + false_neg)) * 100.0
        if ((true_pos + false_pos) == 0):
            prec = 0
        else:
            prec = (true_pos / (true_pos + false_pos)) * 100.0

    else:
        true_pos = len(np.where(cls_labels == 1000)[0])
        false_pos = len(cls) - true_pos
        false_neg = (total_fail) - true_pos
        # print(len(cls), total_fail,true_pos, false_pos, false_neg)
        if ((true_pos + false_neg) == 0):
            recall = 0
        else:
            recall = (true_pos / (true_pos + false_neg)) * 100.0
        if ((true_pos + false_pos) == 0):
            prec = 0
        else:
            prec = (true_pos / (true_pos + false_pos)) * 100.0

    return (coverage, prec, recall)


def describe_invariants_all_labels(all_invariants, layer, fingerprints_tr, fingerprints_tst, labels, labels_test,
                                   ALL=False, Threshold=60, MIS=True, Top=False):
    if (Top == True):
        print("PRINTING RULES WITH HIGHEST RECALL FOR CORRECT CLASSIFICATION TO EVERY LABEL.")
    elif (ALL == True):
        print("PRINTING ALL RULES FOR CORRECT CLASSIFICATION FOR EVERY LABEL.")
    else:
        print("PRINTING RULES FOR CORRECT CLASSIFICATION WITH TRAIN RECALL >= ", Threshold, "%.")

    if (MIS == True):
        if (Top == True):
            print("PRINTING RULES WITH HIGHEST RECALL FOR INCORRECT CLASSIFICATION.")
        else:
            print("PRINTING ALL RULES FOR INCORRECT CLASSIFICATION.")

    cls_list = []
    rules_list = []
    tr_recall_list = []
    tst_prec_list = []
    tst_recall_list = []
    final_df = []

    for cl, invs in all_invariants.items():
        if (cl == -1):
            print("impure:")
            print(invs[0])
            continue

        if (MIS == False and (cl == 1000)):
            continue

        for indx in range(0, len(invs)):
            if (Top and indx > 0):
                continue

            inv = invs[indx]

            neurons = inv[0]
            signature = inv[1]

            if len(neurons) == len(signature):
                VAL = False
            else:
                VAL = True

            tr_suffixes = []
            tst_suffixes = []

            if len(fingerprints_tr) > 0:
                tr_suffixes = fingerprints_tr[layer - 1]
            if len(fingerprints_tst) > 0:
                tst_suffixes = fingerprints_tst[layer - 1]

            tr_recall = 0
            tr_prec = 100
            tst_recll = 0
            tst_prec = 0

            if (len(tr_suffixes) > 0):
                (tr_cov, tr_prec, tr_recall) = calc_prec_recall(tr_suffixes, labels, neurons, signature, cl, VAL,
                                                                supp=inv[2])

            if (len(tst_suffixes) > 0):
                (tst_cov, tst_prec, tst_recall) = calc_prec_recall(tst_suffixes, labels_test, neurons, signature,
                                                                   cl, VAL)

            text = ""
            rule_sig = ""
            if (cl != 1000):
                text = "Rule for correct classification to " + str(cl)
                if (ALL == False) and (Top == False) and (tr_recall < Threshold):
                    continue
                rule_sig = "LAYER:" + str(layer) + ", NEURONS:" + str(inv[0]) + ", " + "SIGNATURE:" + str(
                    inv[1]) + ", " + "SUPPORT:" + str(inv[2])
            # print("Rule:(neurons:",inv[0],",signature:",inv[1],"), Support:",inv[2])

            else:
                text = "Rule for incorrect classification"
                rule_sig = "LAYER:" + str(layer) + ", NEURONS:" + str(inv[0]) + ", " + "SIGNATURE:" + str(
                    inv[1]) + ", " + "SUPPORT:" + str(inv[2])
            # print("Rule:(neurons:",inv[0],",signature:",inv[1],"), Support:",inv[2])

            if (len(tst_suffixes) > 0):
                data = {"Rule Type": [text],
                        "Rule": [rule_sig],
                        "Train Coverage (%)": [tr_cov],
                        "Train Precision (%)": [tr_prec],
                        "Train Recall (%)": [tr_recall],
                        "Test Coverage (%)": [tst_cov],
                        "Test Precision (%)": [tst_prec],
                        "Test Recall (%)": [tst_recall]}

                cls_list.append(cl)
                rules_list.append(inv)
                tr_recall_list.append(tr_recall)
                tst_prec_list.append(tst_prec)
                tst_recall_list.append(tst_recall)

            else:
                data = {"Rule Type": [text],
                        "Rule": [rule_sig],
                        "Train Coverage (%)": [tr_cov],
                        "Train Precision (%)": [tr_prec],
                        "Train Recall (%)": [tr_recall]}
                cls_list.append(cl)
                rules_list.append(inv)
                tr_recall_list.append(tr_recall)

            pd.set_option('display.max_colwidth', None)
            df = pd.DataFrame(data)
            final_df.append(df)
            #styled_df = df.style.hide(axis="index")
            #styled_df1 = styled_df.set_properties(**{
            #    'font-size': '8pt',
            #})

            #display(styled_df1)
            #print(styled_df1)

    df = pd.concat(final_df) if len(final_df) > 0 else None
    return cls_list, rules_list, tr_recall_list, tst_prec_list, tst_recall_list, df


def impure_rules(all_invariants):
    for cl, invs in all_invariants.items():
        if cl == -1:
            print("impure:")
            print(invs[0])
            continue
