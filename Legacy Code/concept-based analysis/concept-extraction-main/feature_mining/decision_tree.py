# Library for decision tree learning
from sklearn import tree
from tqdm import tqdm
import operator
import numpy as np


def get_decision_path(estimator, inp):
  # Extract the decision path taken by an input as an ordered list of indices
  # of the neurons that were evaluated.
  # See: http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
  n_nodes = estimator.tree_.node_count
  feature = estimator.tree_.feature

  # First let's retrieve the decision path of each sample. The decision_path
  # method allows to retrieve the node indicator functions. A non zero element of
  # indicator matrix at the position (i, j) indicates that the sample i goes
  # through the node j.
  X_test = [inp]
  node_indicator = estimator.decision_path(X_test)
  # Similarly, we can also have the leaves ids reached by each sample.
  leaf_id = estimator.apply(X_test)
  # Now, it's possible to get the tests that were used to predict a sample or
  # a group of samples. First, let's make it for the sample.
  node_index = node_indicator.indices[node_indicator.indptr[0]:
                                      node_indicator.indptr[1]]
  neuron_ids = []
  for node_id in node_index:
    if leaf_id[0] == node_id:
        continue
    neuron_ids.append(feature[node_id])
  return neuron_ids

def get_suffix_cluster(neuron_ids, neuron_sig,suffixes):
  # Get the cluster of inputs that such that all inputs in the cluster
  # have provided on/off signature for the provided neurons.
  #
  # The returned cluster is an array of indices (into mnist.train.images).
  return np.where((suffixes[:, neuron_ids] == neuron_sig).all(axis=1))[0]

def get_suffix_cluster_vals(neuron_ids, neuron_sig,suffixes):
  # Get the cluster of inputs that such that all inputs in the cluster
  # have provided on/off signature for the provided neurons.
  #
  # The returned cluster is an array of indices (into mnist.train.images).
  cls_list = []
  for inp_indx in range(0,len(suffixes)):
    val = suffixes[inp_indx]
    neu_indx = 0
    fnd = 1
    for indx in range(0,len(neuron_sig)):
        if (indx %2 == 0):
            oper = neuron_sig[indx]
            continue
        thres = neuron_sig[indx]
        neu = neuron_ids[neu_indx]
        neu_indx = neu_indx + 1
        if (((oper =='<=') and (val[neu] > thres)) or ((oper == '>') and (val[neu] <= thres))):
            fnd = 0
            break
            
    if (fnd == 1):
        cls_list.append(inp_indx)
            

  return cls_list
  
def is_consistent_cluster(cluster, predictions):
  # Check if all inputs within the cluster have the same prediction.
  # 'cluster' is an array of input ids.
  pred = predictions[cluster[0]]
  for i in cluster:
    if predictions[i] != pred:
      return False
  return True

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
  paths =  get_all_paths_rec(0)
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


def get_all_invariants_vals(estimator):
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
    #paths = [[[feature[node]] + p[0], [0] + p[1], p[2], p[3]] for p in get_all_paths_rec(left_child(node))]
    #paths += [[[feature[node]] + p[0], [1] + p[1], p[2], p[3]] for p in get_all_paths_rec(right_child(node))]
    paths = [[[feature[node]] + p[0],['<='] + [threshold[node]] + p[1], p[2], p[3]] for p in get_all_paths_rec(left_child(node))]
    paths += [[[feature[node]] + p[0],['>'] + [threshold[node]] + p[1], p[2], p[3]] for p in get_all_paths_rec(right_child(node))]
    return paths
  paths =  get_all_paths_rec(0)
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
  
def check_pattern_inter(layer_vals,suff,neuron_ids,neuron_sig,VAL = True,ALL=False):
   
  if (VAL == False):
    if ((suff[:,neuron_ids][0] == neuron_sig).all(axis=0)):
      return True
    else:
      return False

  found = True
  oper = -1
  layer_vals = (layer_vals).flatten()
   
  for ix in range(0,len(neuron_ids)):
    found = True
    for ind in range(0,len(neuron_ids[ix])):
      if (ind % 2 == 0):
        op = neuron_sig[ix][ind]
        if (op == '<='):
          oper = 0
        else:
          oper = 1
      else:
        v = neuron_ids[ix][ind]
        vsig = neuron_sig[ix][ind]
        val = layer_vals[v]
        #print(oper, v, vsig, val)
        if (oper == 0):
          if (val > vsig):
            found = False
            break
        else:
          if (val <= vsig):
            found = False
            break
        oper = -1
    if (found == True):
      break 

  if (found == False):
    return -1
  else:
    return ix 

def validate(invariant, label_trgt, activation_vectors, labels_gt):
  # Returns the precision and recall for the given invariant; label_trgt is the label associated with the invariant
  # invariant is in the form of [neuron_ids, neuron_sig, nsamples] triple
  # labels_gt are the ground truth labels    
  neurons = []
  for indx in range(0,len(invariant[0])):
      neurons.append(-1)
      neurons.append(invariant[0][indx])

  prec_recall_num = 0
  recall_denom = 0
  prec_denom = 0
  for indx1 in range(0,len(activation_vectors)):
    label_pred = label_trgt
    label_gt = labels_gt[indx1]
    match = check_pattern_inter(activation_vectors[indx1],activation_vectors[indx1],[neurons],[invariant[1]])

    if label_gt == label_trgt:
      recall_denom += 1
    if (match != -1):
      prec_denom += 1
      if label_pred == label_gt:
        prec_recall_num += 1                         
  
  if (recall_denom == 0):
      print("CLASS;", label_trgt, invariant, ";Precision:",0,";Recall:",0)
  else:
      if prec_denom == 0:
          print("CLASS;", label_trgt, invariant, ";Precision:",0,";Recall:", float(prec_recall_num/recall_denom)*100)
      else:
          print("CLASS;", label_trgt, invariant, ";Precision:",float(prec_recall_num/prec_denom)*100,";Recall:", float(prec_recall_num/recall_denom)*100)

def get_invariant(activation_vectors, labels):
    # Generate Decision Trees 
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(activation_vectors, labels)
    all_invariants = get_all_invariants_vals(dtree)
    return all_invariants

def get_tree(activation_vectors, labels):
    # Generate Decision Trees 
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(activation_vectors, labels)
    return dtree

def validate_n_invariants(invariants, label_trgt, activation_vectors, labels_gt):
  # Returns the precision and recall for the given invariant; label_trgt is the label associated with the invariant
  # invariant is in the form of [neuron_ids, neuron_sig, nsamples] triple
  # labels_gt are the ground truth labels    
  prec_recall_num = 0
  recall_denom = 0
  prec_denom = 0
  for indx1 in range(0,len(activation_vectors)):
    label_pred = label_trgt
    label_gt = labels_gt[indx1]
    match = -1
    for indx2 in range(0, len(invariants)):
      neurons = []
      for indx in range(0,len(invariants[indx2][0])):
          neurons.append(-1)
          neurons.append(invariants[indx2][0][indx])
      match = check_pattern_inter(activation_vectors[indx1],activation_vectors[indx1],[neurons],[invariants[indx2][1]])
      if (match != -1):
        break
    
    if label_gt == label_trgt:
      recall_denom += 1
    if (match != -1):
      prec_denom += 1
      if label_pred == label_gt:
        prec_recall_num += 1                         
  
  if (recall_denom == 0):
      print("CLASS;", label_trgt, ";Precision:",0,";Recall:",0)
      return 0, 0
  else:
      if prec_denom == 0:
          print("CLASS;", label_trgt, ";Precision:",0,";Recall:", float(prec_recall_num/recall_denom)*100)
          return 0, float(prec_recall_num/recall_denom)*100
      else:
          print("CLASS;", label_trgt, ";Precision:",float(prec_recall_num/prec_denom)*100,";Recall:", float(prec_recall_num/recall_denom)*100)
          return float(prec_recall_num/prec_denom)*100, float(prec_recall_num/recall_denom)*100