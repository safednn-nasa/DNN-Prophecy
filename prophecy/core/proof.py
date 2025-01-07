import os

path = os.environ['PATH']
print(path)
print(os.environ['PATH'])
os.environ['PATH'] = path + ':/ProphecyPlus/Marabou/Marabou_bld:/ProphecyPlus/Marabou/Marabou_bld/build:/ProphecyPlus/Marabou/Marabou_bld/build/bin'
print(os.environ['PATH'])

def check_pattern(layer_vals: list, neuron_ids: list, neuron_sig: list) -> bool:
    """
        Check if the provided layer values satisfy the provided neuron signature.
    :param layer_vals:
    :param neuron_ids:
    :param neuron_sig:
    :return:
    """
    found = True
    oper = -1
    # layer_vals = (layer_vals).flatten()

    for ind in range(0, len(neuron_sig)):
        if ind % 2 == 0:
            op = neuron_sig[ind]
            if op == '<=':
                oper = 0
            else:
                oper = 1
        else:
            v = int(neuron_ids[(int)(ind / 2)])
            vsig = float(neuron_sig[ind])
            val = float(layer_vals[v])
            # print(v,vsig,val,oper)
            if oper == 0:
                if val > vsig:
                    # print(v,val,vsig,oper)
                    found = False
                    break
            else:
                if val <= vsig:
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

##################################################################
#from keras import backend
#func3 = None
#for layer in model.layers:
#    print(layer.name)
#    if (layer.name == 'relu_3'):
#      func3 = backend.function(model.input, [layer.output])
#fingerprint_3 = []
#if (func3 != None):
#  fingerprint_3 = func3(x_train_rshape)

################################################################
#print(len(x_train))
#x_train_min = np.zeros(5)
#x_train_max = np.zeros(5)

#for indx in range(0,5):
#  x_train_min[indx] = np.min(x_train[:,indx])
#  x_train_max[indx] = np.max(x_train[:,indx])

#print(x_train_min)
#print(x_train_max)


#print("RULE:")
#rule_neurons_list_3 = []
#rule_neurons = (relu_3_rule_neurons.array[0]).split(",")
#for indx in range(0, len(rule_neurons)):
#    rule_neurons[indx] = (rule_neurons[indx]).strip()
#    rule_neurons[indx] = (rule_neurons[indx]).replace("[", "")
#    rule_neurons[indx] = (rule_neurons[indx]).replace("]","")
#    rule_neurons_list_3.append(int(rule_neurons[indx]))

#print(rule_neurons_list_3)

#rule_sig_list_3 = []
#rule_sig = (relu_3_rule_signature.array[0]).split(",")
#for indx in range(0, len(rule_sig)):
#    rule_sig[indx] = (rule_sig[indx]).strip()
#    rule_sig[indx] = (rule_sig[indx]).replace("[", "")
#    rule_sig[indx] = (rule_sig[indx]).replace("]","")
#    rule_sig_list_3.append(int(rule_sig[indx]))

#print(rule_sig_list_3)

#fngprnt = (fingerprint_3[0] > 0.0).astype('int')
#indices = get_suffix_cluster(rule_neurons_list_3, rule_sig_list_3, fngprnt)
#print("indices:", len(indices))

#x_train3 = []
#fngprnt3 = []
#for indx in range(0, len(indices)):
#    if (indx == 0):
#      print("INP:",x_train[indices[indx]])
#      print("FINGER:",fingerprint_3[0][indices[indx]])
#    x_train3.append(x_train[indices[indx]])
#    fngprnt3.append(fingerprint_3[0][indices[indx]])

#x_train3 = np.array(x_train3)
#fngprnt3 = np.array(fngprnt3)

#print(np.shape(x_train3))
#x_train_min3 = np.zeros(5)
#x_train_max3 = np.zeros(5)

#for indx in range(0,5):
#  x_train_min3[indx] = np.min(x_train3[:,indx])
#  x_train_max3[indx] = np.max(x_train3[:,indx])

#print(x_train_min3)
#print(x_train_max3)

#print(np.shape(fngprnt3))
#fngprnt_min3 = np.zeros(len(fngprnt3[0]))
#fngprnt_max3 = np.zeros(len(fngprnt3[0]))

#for indx in range(0,len(fngprnt3[0])):
#  fngprnt_min3[indx] = np.min(fngprnt3[:,indx])
#  fngprnt_max3[indx] = np.max(fngprnt3[:,indx])

#print(fngprnt_min3)
#print(fngprnt_max3)





