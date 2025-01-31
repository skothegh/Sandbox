import numpy as np

import heapq as hq

from graph_traversal import dijkstra, bfs

def power_iteration(G,epsilon):
    pass

def page_rank(A,beta=0.9,epsilon=1e-5):
    '''
    can i do this more elegant? or is everything here needed?
    '''
    M = A.T
    L = M.shape[0]#len(adj_dict.keys())
    r0 = jnp.ones(L)/L

    for i in range(L):
        if(M[:,i].sum() == 0):
            M = M.at[:,i].set(1/L)
        else:
            M = M.at[:,i].divide(M[:,i].sum())

    G = beta * M + (1-beta)*(1/L)

    c = 10
    while(c > epsilon):
        r = G@r0
        c = jnp.abs(r - r0).sum()
        r0 = r

    return r


def brandes(nodes,directed=False):
    dir = 2 # take care of double-counting in undirected graphs
    if(directed): dir = 1

    Q = []
    hq.heapify(Q)
    cb = {} # betweeness_centrality

    for n in nodes.keys():
        cb[n] = 0

    for s in nodes.keys():
        ds = {} #dependency of s on all others
        dist = dijkstra(nodes,s,verbose=False)

        for n in dist:
            ds[n] = 0
            hq.heappush(Q,(1/(dist[n]["distance"] + 1e-10),n))

        while(len(Q)):
            wd, wn =  hq.heappop(Q) #1/distance, id
            for v in nodes[wn].next:
                if(dist[wn]["distance"] == dist[v]["distance"] + 1): # v€P_s[w]
                    ds[v] += (dist[v]["num_paths"]/dist[wn]["num_paths"])*(1 + ds[wn])
            if(not wn is s):
                cb[wn] += ds[wn]/2

    return cb
