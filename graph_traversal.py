import numpy as np
import heapq as hq


def dijkstra(nodes,source,verbose=True):
    '''
    for weighted graphs, needs (distance,node) tuple in heap
    '''
    order = []
    queue = [] # gonna be a heap (distance,node) "unvisited queue"
    traversal = {} # the dictionary containing the information about everything
    for n in nodes.keys():
        traversal[n] = {}
        traversal[n]["distance"] = np.inf
        traversal[n]["parent"] = []
        traversal[n]["visited"] = False
        traversal[n]["num_paths"] = 1

    traversal[source]["distance"] = 0
    queue.append( (0,source) )
    hq.heapify(queue)

    while(len(queue)):
        u = hq.heappop(queue)
        if(not traversal[u[1]]["visited"]):
            order.append(u[1])
            traversal[u[1]]["visited"] = True

            for v in nodes[u[1]].next:
                alt = traversal[u[1]]["distance"] + nodes[u[1]].next[v]
                if(alt == traversal[v]["distance"]):
                    traversal[v]["parent"].append(u[1])
                    traversal[v]["num_paths"] += traversal[u[1]]["num_paths"]

                elif(alt < traversal[v]["distance"]):
                    traversal[v]["parent"] = [u[1]]
                    traversal[v]["distance"] = alt
                    traversal[v]["num_paths"] = traversal[u[1]]["num_paths"]

                    hq.heappush(queue,(alt,v))
    if(verbose):
        return traversal, order
    else:
        return traversal


def dijkstra_target(nodes,source,target):
    '''
    stops when the targst node is selected as "current node",
    i.e. is popped off the heap

    then creates and returns the shortest path between source and target
    '''
    order = []
    queue = [] # gonna be a heap (distance,node) "unvisited queue"
    traversal = {} # the dictionary containing the information about everything
    for n in nodes.keys():
        traversal[n] = {}
        traversal[n]["distance"] = np.inf
        traversal[n]["parent"] = []
        traversal[n]["visited"] = False
        traversal[n]["num_paths"] = 1

    traversal[source]["distance"] = 0
    queue.append( (0,source) )
    hq.heapify(queue)

    while(len(queue)):
        u = hq.heappop(queue)
        if(u[1] == target):
            break

        if(not traversal[u[1]]["visited"]):
            order.append(u[1])
            traversal[u[1]]["visited"] = True

            for v in nodes[u[1]].next:
                alt = traversal[u[1]]["distance"] + nodes[u[1]].next[v]
                if(alt == traversal[v]["distance"]):
                    traversal[v]["parent"].append(u[1])
                    traversal[v]["num_paths"] += traversal[u[1]]["num_paths"]

                elif(alt < traversal[v]["distance"]):
                    traversal[v]["parent"] = [u[1]]
                    traversal[v]["distance"] = alt
                    traversal[v]["num_paths"] = traversal[u[1]]["num_paths"]

                    hq.heappush(queue,(alt,v))

    shortest_path = []
    temp = target
    shortest_path.append(temp)
    while(not temp == source):
            parent_temp = traversal[temp]["parent"][0] # picks one if multiple parents
            shortest_path.append(parent_temp)
            temp = parent_temp


    shortest_path.reverse()
    return shortest_path, len(shortest_path) - 1

def bfs(nodes, source):
    order = []
    queue = []

    traversal = {}
    for n in nodes.keys():
        traversal[n] = {}
        traversal[n]["distance"] = np.inf
        traversal[n]["parents"] = []
        traversal[n]["visited"] = False
        traversal[n]["num_paths"] = 1

    traversal[source]["visited"] = True
    traversal[source]["distance"] = 0

    queue.append(source)
    order.append(source)
    while(len(queue)):
        u = queue.pop()
        traversal[u]["visited"] = True

        for v in nodes[u].next:
            alt = traversal[u]["distance"] + 1

            if(alt == traversal[v]["distance"]):
                traversal[v]["parents"].append(u)
                traversal[v]["num_paths"] += traversal[u]["num_paths"]

            elif(alt < traversal[v]["distance"]):
                order.append(v)
                traversal[v]["parents"] = [u]
                traversal[v]["distance"] = alt
                traversal[v]["num_paths"] = traversal[u]["num_paths"]
                queue.insert(0,v)
    return traversal, order
