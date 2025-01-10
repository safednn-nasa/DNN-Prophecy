import networkx as nx
import numpy as np
import pandas as pd

# Generate the mixing matrix for a given attribute.
# Param:
#     G: a networkx graph
#     attr: str. node attribute for calculating assortivity
#     attr_type_list: False, None, or array-like.
#           a list of attribute values. If None/False, extract and use all values of the attribute
#     weight: str, False, None. if False/None, convert network to binary; else, use that as weight
# return:
#    out_df: Pandas DataFrame. A mixing matrix with attribute list as row and column names.

def mixing_matrix(G,
                  attr,
                  attr_type_list=False,
                  weight = "weight", 
                  ): 
    node_df = pd.DataFrame(list(G.nodes(data=attr)), columns=["node","attr"])
    nodelist = node_df["node"]
    
    if weight:
        adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight).todense()
    else:
        adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=None).todense()
    
    if attr_type_list:
        total_types = attr_type_list
    else:
        total_types = set(nx.get_node_attributes(G, attr).values())
    
    W = adj.sum()
    out = np.zeros((len(total_types),len(total_types)))
    for i, type1 in enumerate(total_types):
        for j, type2 in enumerate(total_types):
            node_list1 = node_df.loc[node_df["attr"] == type1].index
            node_list2 = node_df.loc[node_df["attr"] == type2].index
            out[i,j] = adj[np.ix_(node_list1, node_list2)].sum() / W
            
    out_df = pd.DataFrame(out, columns=total_types, index=total_types)
    return out_df

# Calculate the assortivity for a discrete variable
# Param:
    # G: a networkx graph
    # attr: str. node attribute for calculating assortivity. Must be a categorical attribute
    # attr_type_list: False, None, or array-like.
    #       a list of attribute values. If None/False, extract and use all values of the attribute
    # weight: str, False, None. if False/None, convert network to binary; else, use that as weight
    # SE: Boolean. whether to calculate jackknife standard error
    # M: int. Parameter for calculating SE
# return:
    # r: Assortivity coefficient
    # se: Jackknife standard error

def assortment_discrete(G,
                        attr,
                        attr_type_list=False,
                        weight = "weight",
                        SE = False,
                        M = 1
                       ):
    # A helper function

    def gen_out(adj, total_types, node_df):
        W = adj.sum()
        out = np.zeros((len(total_types),len(total_types)))
        for i, type1 in enumerate(total_types):
            for j, type2 in enumerate(total_types):
                node_list1 = node_df.loc[node_df["attr"] == type1].index
                node_list2 = node_df.loc[node_df["attr"] == type2].index
                out[i,j] = adj[np.ix_(node_list1, node_list2)].sum() / W
        return out

    node_df = pd.DataFrame(list(G.nodes(data=attr)), columns=["node","attr"])
    nodelist = node_df["node"]
    
    if weight:
        adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight).todense()
    else:
        adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=None).todense()
    
    if attr_type_list:
        total_types = attr_type_list
    else:
        total_types = set(nx.get_node_attributes(G, attr).values())
    
    out = gen_out(adj, total_types, node_df)
            
    numer = np.diag(out).sum() - (out.sum(axis=0) * out.sum(axis=1)).sum()
    denom = 1 - (out.sum(axis=0) * out.sum(axis=1)).sum()
    r = numer / denom
    
    if SE:
        se = 0
        N = np.where(adj > 0)
        E = np.arange(0, len(N[0])+1, M)

        if E[-1] < len(N[0]):
            E = np.append(E, len(N[0]))

        for g in range(len(E)-1):
            adj2 = adj.copy()
            adj2[N[0][E[g]:E[g+1]], N[1][E[g]:E[g+1]]] = 0
            out2 = gen_out(adj2, total_types, node_df)
            numer = np.diag(out2).sum() - (out2.sum(axis=0) * out2.sum(axis=1)).sum()
            denom = 1 - (out2.sum(axis=0) * out2.sum(axis=1)).sum()
            ri = numer / denom
            se += (ri - r) ** 2
        return r, se
    
    return r

# Calculate the assortivity for a discrete variable
# Param:
    # G: a networkx graph
    # attr: str. node attribute for calculating assortivity. Must be a numeric attribute
    # attr_type_list: False, None, or array-like.
    #       a list of attribute values. If None/False, extract and use all values of the attribute
    # weight: str, False, None. if False/None, convert network to binary; else, use that as weight
    # SE: Boolean. whether to calculate jackknife standard error
    # M: int. Parameter for calculating SE
# return:
    # r: Assortivity coefficient
    # se: Jackknife standard error
    
def assortment_continuous(G,
                        attr,
                        weight = "weight",
                        SE = False,
                        M = 1
                        ):
    node_df = pd.DataFrame(list(G.nodes(data=attr)), columns=["node","attr"])
    node_df["attr"] = pd.to_numeric(node_df["attr"])
    nodelist = node_df["node"]
    attrlist = node_df["attr"]

    if weight:
        adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight).todense()
    else:
        adj = nx.adjacency_matrix(G, nodelist=nodelist, weight=None).todense()
        
    ji = np.expand_dims(attrlist.to_numpy(), axis=0)
    ki = np.expand_dims(attrlist.to_numpy(), axis=1)
    top1 = ((adj @ ki).T @ ji.T).sum()
    top2 = (adj.T @ ji.T).sum()
    top3 = (adj @ ki).sum()
    bottom1 = (adj.T @ (ji ** 2).T).sum()
    bottom2 = (adj.T @ ji.T).sum() ** 2
    bottom3 = (adj @ (ki ** 2)).sum()
    bottom4 = (adj @ ki).sum() ** 2
    W = adj.sum()
    numer = top1 - 1/W * (top2 * top3)
    denom = np.sqrt((bottom1-1/W * bottom2) * (bottom3-1/W * bottom4))
    r = numer/denom
    
    if SE:
        se = 0
        N = np.where(adj > 0)
        E = np.arange(0, len(N[0])+1, M)

        if E[-1] < len(N[0]):
            E = np.append(E, len(N[0]))

        for g in range(len(E)-1):
            adj2 = adj.copy()
            adj2[N[0][E[g]:E[g+1]], N[1][E[g]:E[g+1]]] = 0
            top1 = ((adj2 @ ki).T @ ji.T).sum()
            top2 = (adj2.T @ ji.T).sum()
            top3 = (adj2 @ ki).sum()
            bottom1 = (adj2.T @ (ji ** 2).T).sum()
            bottom2 = (adj2.T @ ji.T).sum() ** 2
            bottom3 = (adj2 @ (ki ** 2)).sum()
            bottom4 = (adj2 @ ki).sum() ** 2
            W = adj2.sum()
            numer = top1 - 1/W * (top2 * top3)
            denom = np.sqrt((bottom1-1/W * bottom2) * (bottom3-1/W * bottom4))
            ri = numer / denom
            se += (ri - r) ** 2
        return r, se
    
    return r