import networkx as nx
from nltk.app.nemo_app import images

from graph import *
from priority_queue import *
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from collections import defaultdict
# Global Constants
HEIGHT = 600
WIDTH = 1280
animate_time = 0.5
# Global Variables
G_di = nx.DiGraph()
G = nx.Graph()
G_sub = Graph()
fromNodeString = ""
toNodeString = ""
weight = ""
heuristic = ''
node = ''
heuristic_dic = {}
nodes_index = {}
pos = {}

goal_nodes = []
start_node = ""
color_map = []
visited_nodes = []
verbose = False
time_sleep = 1
index = 0

eVisited = []
ePath = []
animatedNodes = []
inputList=[]

class Graphical:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DFSUtil(self, v, visited,inputs):

        visited.add(v)
        print(v, end=' ')
        inputs.append(v)

        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited,inputs)
    def DFS(self, v,inputs):

        visited = set()
        self.DFSUtil(v, visited,inputs)
        return inputs

def fromNode(userFromNodeString):
    global fromNodeString
    fromNodeString = userFromNodeString


def toNode(userToNodeString):
    global toNodeString
    toNodeString = userToNodeString


def addEdgeToGraph():
    global fromNodeString, toNodeString, weight, G_di, fromNodeField, toNodeField, weightField, console
    fromNodeString = fromNodeField.get()
    toNodeString = toNodeField.get()
    weight = weightField.get()
    print("From Node " + fromNodeString + " to Node " + toNodeString + " with weight " + weight)
    console.configure(text="Edge Added : From Node " + fromNodeString + " to Node " + toNodeString + " with weight " + weight)
    G_di.add_weighted_edges_from([(fromNodeString, toNodeString, int(weight))])
    G.add_weighted_edges_from([(fromNodeString, toNodeString, int(weight))])
    G_sub.addNode(fromNodeString)
    G_sub.addNode(toNodeString)
    G_sub.connect(fromNodeString, toNodeString, int(weight))
    nodes_index[fromNodeString] = index
    inputList.append(fromNodeField.get())
    inputList.append(toNodeField.get())


def addWeight(weightString):
    global weight
    weight = int(weightString)


def addNode(NodeString):
    global node
    node = NodeString


def addHeuristic():
    global heuristic, heuristic_dic, node,  nodes_index, index, NodeField, HeuristicValueField
    heuristicString = HeuristicValueField.get()
    node = NodeField.get()
    heuristic = int(heuristicString)
    heuristic_dic[node] = heuristic
    nodes_index[node] = index
    index = index + 1
    print("Heuristic value of " + node + " is " + heuristicString)
    console.configure(text="Heuristic Added : value of " + node + " is " + heuristicString)


def addStartNode(startString):
    global start_node
    start_node = startString



def addGoalNode():
    global start_node, goal_nodes
    start_node = startField.get()
    goalNodesString = goalField.get()
    goal_nodes = goalNodesString.split(',')
    console.configure(text="Start and Goal/s Added : Start = " + start_node + ", Goal/s = " + str(goal_nodes))


def drawDiGraph():
    global G_di, goal_nodes, start_node, color_map, a, canvas, f, pos, console
    color()
    labels = nx.get_edge_attributes(G_di, 'weight')
    a = f.add_subplot(111)
    pos = nx.spring_layout(G_di)
    nx.draw_networkx_nodes(G_di, pos, node_size=500, node_color=color_map, ax=a)
    nx.draw_networkx_edges(G_di, pos, edgelist=G_di.edges(), edge_color='black', ax=a)
    nx.draw_networkx_edge_labels(G_di, pos, edge_labels=labels, ax=a)
    nx.draw_networkx_labels(G_di, pos, ax=a)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)
    console.configure(text="Directed Graph Drawn")


def color():
    global start_node, goal_nodes
    color_map.clear()
    for noder in G.nodes():
        if noder == start_node:
            color_map.append('red')
        elif noder in goal_nodes:
            color_map.append('green')
        else:
            color_map.append('grey')


def aniColor():
    global start_node, goal_nodes, animatedNodes, color_map
    color_map.clear()
    for noder in G.nodes():
        if noder in animatedNodes:
            color_map.append('#34c3eb')
        elif noder in goal_nodes:
            color_map.append('green')
        elif noder == start_node:
            print(noder)
            color_map.append('red')
        else:
            color_map.append('grey')


def drawGraph():
    global G, goal_nodes, start_node, color_map, a, canvas, f, pos, console
    color()
    labels = nx.get_edge_attributes(G, 'weight')
    a = f.add_subplot(111)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=color_map, ax=a)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', ax=a)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=a)
    nx.draw_networkx_labels(G, pos, ax=a)
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)
    console.configure(text="Undirected Graph Drawn")


def animateG(eVisited, ePath):
    global G, a, pos, f, canvas, animate_time
    a = f.add_subplot(111)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=color_map, ax=a)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', ax=a)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=a)
    nx.draw_networkx_labels(G, pos, ax=a)
    # nx.draw_networkx_edges(G, pos, edgelist=eVisited, width=6, edge_color="#7be016", ax=a)
    nx.draw_networkx_edges(G, pos, edgelist=ePath, width=6, edge_color="#7be016", ax=a)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)
    canvas.get_tk_widget().update()
    time.sleep(animate_time)


def animateNode():
    global G, a, pos, f, canvas, color_map, animate_time
    a = f.add_subplot(111)
    labels = nx.get_edge_attributes(G, 'weight')
    aniColor()
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=color_map, ax=a)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', ax=a)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=a)
    nx.draw_networkx_labels(G, pos, ax=a)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=color_map, ax=a)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)
    canvas.get_tk_widget().update()
    time.sleep(animate_time)


def animateNodeDir():
    global G_di, a, pos, f, canvas, color_map, animate_time
    a = f.add_subplot(111)
    labels = nx.get_edge_attributes(G_di, 'weight')
    aniColor()
    nx.draw_networkx_nodes(G_di, pos, node_size=500, node_color=color_map, ax=a)
    nx.draw_networkx_edges(G_di, pos, edgelist=G_di.edges(), edge_color='black', ax=a)
    nx.draw_networkx_edge_labels(G_di, pos, edge_labels=labels, ax=a)
    nx.draw_networkx_labels(G_di, pos, ax=a)
    nx.draw_networkx_nodes(G_di, pos, node_size=500, node_color=color_map, ax=a)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)
    canvas.get_tk_widget().update()
    time.sleep(animate_time)


def animateGDir(eVisited, ePath):
    global G_di, a, pos, f, canvas, animate_time
    a = f.add_subplot(111)
    labels = nx.get_edge_attributes(G_di, 'weight')
    nx.draw_networkx_nodes(G_di, pos, node_size=500, node_color=color_map, ax=a)
    nx.draw_networkx_edges(G_di, pos, edgelist=G_di.edges(), edge_color='black', ax=a)
    nx.draw_networkx_edge_labels(G_di, pos, edge_labels=labels, ax=a)
    nx.draw_networkx_labels(G_di, pos, ax=a)
    # nx.draw_networkx_edges(G_di, pos, edgelist=eVisited, width=6, edge_color="#7be016", ax=a)
    nx.draw_networkx_edges(G_di, pos, edgelist=ePath, width=6, edge_color="#7be016", ax=a)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)
    canvas.get_tk_widget().update()
    time.sleep(animate_time)





def bfs():
    console.configure(text='BFS Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, animatedNodes
    visited = {node: False for node in G.nodes}
    visited_string = ''
    queue = [start_node]
    animatedNodes = []
    visited[start_node] = True
    result = []
    eVisited = []  # edges which have been considered
    ePath = []
    path = {start_node: start_node}
    while queue:
        cur_node = queue.pop(0)
        result.append(cur_node)
        animatedNodes.append(cur_node)
        animateNode()
        visited_nodes.append(cur_node)
        for goal in goal_nodes:
            if cur_node == goal:
                visited_string = visited_string + ' , '.join(result)
                console.configure(text="Visited Nodes are : " + visited_string)
                console.update()
                curr_goal = goal
                actual_path_nodes = [curr_goal]
                while curr_goal != start_node:
                    curr_goal = path[curr_goal]
                    actual_path_nodes.insert(0, curr_goal)
                for i in range(len(actual_path_nodes)):
                    if actual_path_nodes[i] != goal_nodes[0]:
                        ePath.append((actual_path_nodes[i], actual_path_nodes[i+1]))
                        animateG(eVisited, ePath)
                console.configure(text="Visited Nodes are : " + visited_string + "\nActual Path is " + str(actual_path_nodes))
                return
        for node in G.neighbors(cur_node):
            if not visited[node]:
                queue.append(node)
                path[node] = cur_node
                eVisited.append((cur_node, node))
                animateG(eVisited, ePath)
                visited[node] = True
    visited_string = visited_string + ' , '.join(result)
    console.configure(text="Goal not found  , Visited Nodes are : " + visited_string)


def bfsDirected():
    console.configure(text='BFS Algorithm started for directed graph.')
    global start_node, visited_nodes, G_di, animatedNodes
    visited = {node: False for node in G_di.nodes}
    visited_string = ''
    queue = [start_node]
    visited[start_node] = True
    result = []
    eVisited = []  # edges which have been considered
    ePath = []
    animatedNodes = []
    path = {start_node: start_node}
    while queue:
        cur_node = queue.pop(0)
        result.append(cur_node)
        animatedNodes.append(cur_node)
        animateNodeDir()
        visited_nodes.append(cur_node)
        for goal in goal_nodes:
            if cur_node == goal:
                visited_string = visited_string + ' , '.join(result)
                console.configure(text="Visited Nodes are : " + visited_string)
                console.update()
                curr_goal = goal
                actual_path_nodes = [curr_goal]
                while curr_goal != start_node:
                    curr_goal = path[curr_goal]
                    actual_path_nodes.insert(0, curr_goal)
                for i in range(len(actual_path_nodes)):
                    if actual_path_nodes[i] != goal_nodes[0]:
                        ePath.append((actual_path_nodes[i], actual_path_nodes[i + 1]))
                        animateGDir(eVisited, ePath)
                console.configure(
                    text="Visited Nodes are : " + visited_string + "\nActual Path is " + str(actual_path_nodes))
                return
        for node in G_di.successors(cur_node):
            if not visited[node]:
                queue.append(node)
                path[node] = cur_node
                eVisited.append((cur_node, node))
                animateGDir(eVisited, ePath)
                visited[node] = True
    visited_string = visited_string + ' , '.join(result)
    console.configure(text="Goal not found  , Visited Nodes are : " + visited_string)


def dfs():
    console.configure(text='DFS Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, animatedNodes
    visited = {node: False for node in G.nodes}
    visited_string = ''
    queue = [start_node]
    visited[start_node] = True
    result = []
    eVisited = []  # edges which have been considered
    ePath = []
    path = {}
    path[start_node] = start_node
    last_node = start_node
    animatedNodes = []
    while queue:
        cur_node = queue.pop()
        result.append(cur_node)
        animatedNodes.append(cur_node)
        animateNode()
        if last_node != start_node:
            eVisited.append((last_node, cur_node))
            print(eVisited)
            animateG(eVisited, ePath)
            last_node = cur_node
        visited_nodes.append(cur_node)
        for goal in goal_nodes:
            if cur_node == goal:
                visited_string = visited_string + ' , '.join(result)
                console.configure(text="Visited Nodes are : " + visited_string)
                console.update()
                curr_goal = goal
                actual_path_nodes = [curr_goal]
                while curr_goal != start_node:
                    curr_goal = path[curr_goal]
                    actual_path_nodes.insert(0, curr_goal)
                for i in range(len(actual_path_nodes)):
                    if actual_path_nodes[i] != goal_nodes[0]:
                        ePath.append((actual_path_nodes[i], actual_path_nodes[i + 1]))
                        animateG(eVisited, ePath)
                console.configure(
                    text="Visited Nodes are : " + visited_string + "\nActual Path is " + str(actual_path_nodes))
                return
        for node in G.neighbors(cur_node):
            if not visited[node]:
                queue.append(node)
                path[node] = cur_node
                visited[node] = True
    visited_string = visited_string + ' , '.join(result)
    console.configure(text="Goal not found  , Visited Nodes are : " + visited_string)



def dfs_directed():
    console.configure(text='DFS Algorithm started for directed graph.')
    global start_node, G_di, visited_nodes, animatedNodes
    visited = {node: False for node in G_di.nodes}
    visited_string = ''
    queue = [start_node]
    visited[start_node] = True
    result = []
    eVisited = []  # edges which have been considered
    ePath = []
    path = {}
    path[start_node] = start_node
    last_node = start_node
    animatedNodes = []
    while queue:
        cur_node = queue.pop()
        result.append(cur_node)
        animatedNodes.append(cur_node)
        animateNodeDir()
        if last_node != start_node:
            eVisited.append((last_node, cur_node))
            animateGDir(eVisited, ePath)
            last_node = cur_node
        visited_nodes.append(cur_node)
        for goal in goal_nodes:
            if cur_node == goal:
                visited_string = visited_string + ' , '.join(result)
                console.configure(text="Visited Nodes are : " + visited_string)
                console.update()
                curr_goal = goal
                actual_path_nodes = [curr_goal]
                while curr_goal != start_node:
                    curr_goal = path[curr_goal]
                    actual_path_nodes.insert(0, curr_goal)
                for i in range(len(actual_path_nodes)):
                    if actual_path_nodes[i] != goal_nodes[0]:
                        ePath.append((actual_path_nodes[i], actual_path_nodes[i + 1]))
                        animateGDir(eVisited, ePath)
                console.configure(
                    text="Visited Nodes are : " + visited_string + "\nActual Path is " + str(actual_path_nodes))
                return
        for node in G_di.successors(cur_node):
            if not visited[node]:
                queue.append(node)
                path[node] = cur_node
                visited[node] = True
    visited_string = visited_string + ' , '.join(result)
    console.configure(text="Goal not found  , Visited Nodes are : " + visited_string)


def ucs():
    console.configure(text='UCS Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, G_di, goal_nodes, verbose, time_sleep, G_sub, animatedNodes
    # UCS uses priority queue, priority is the cumulative cost (smaller cost)
    queue = PriorityQueue()
    animatedNodes = []
    visited_string = start_node
    # expands initial node
    # get the keys of all successors of initial node
    keys_successors = G_sub.getSuccessors(start_node)
    animatedNodes.append(start_node)
    animateNode()

    # adds the keys of successors in priority queue
    for key_successor in keys_successors:
        weight_inner = G_sub.getWeightEdge(start_node, key_successor)
        # each item of queue is a tuple (key, cumulative_cost)
        queue.insert((key_successor, weight_inner), weight_inner)

    reached_goal, cumulative_cost_goal = False, -1
    while not queue.is_empty():
        # remove item of queue, remember: item of queue is a tuple (key, cumulative_cost)
        key_current_node, cost_node = queue.remove()
        visited_string = visited_string + ' , ' + key_current_node
        animatedNodes.append(key_current_node)
        animateNode()
        for goal in goal_nodes:
            if key_current_node == goal:
                reached_goal, cumulative_cost_goal = True, cost_node
                # visited_string = visited_string + '-->' + key_current_node[0]
                console.configure(
                    text='Visited nodes through search are: ' + visited_string + '\nCumulative Cost: ' + str(
                        cumulative_cost_goal))
                return

        # get all successors of key_current_node
        keys_successors = G_sub.getSuccessors(key_current_node)

        if keys_successors:  # checks if contains successors
            # insert all successors of key_current_node in the queue
            for key_successor in keys_successors:
                cumulative_cost = G_sub.getWeightEdge(key_current_node, key_successor) + cost_node
                queue.insert((key_successor, cumulative_cost), cumulative_cost)

    if reached_goal:
        print('\nReached goal! Cost: %s\n' % cumulative_cost_goal)
    else:
        print('\nUnfulfilled goal.\n')


def ucs_directed():
    global start_node, G, visited_nodes, G_di, goal_nodes, verbose, time_sleep, G_sub, animatedNodes
    console.configure(text='UCS Algorithm started for directed graph.')
    # UCS uses priority queue, priority is the cumulative cost (smaller cost)
    queue = PriorityQueue()
    visited_string = start_node
    animatedNodes = []
    # expands initial node
    # get the keys of all successors of initial node
    keys_successors = G_sub.getSuccessors(start_node)
    animatedNodes.append(start_node)
    animateNodeDir()

    # adds the keys of successors in priority queue
    for key_successor in keys_successors:
        weight_inner = G_sub.getWeightEdge(start_node, key_successor)
        # each item of queue is a tuple (key, cumulative_cost)
        queue.insert((key_successor, weight_inner), weight_inner)

    reached_goal, cumulative_cost_goal = False, -1
    while not queue.is_empty():
        # remove item of queue, remember: item of queue is a tuple (key, cumulative_cost)
        key_current_node, cost_node = queue.remove()
        visited_string = visited_string + ' , ' + key_current_node
        animatedNodes.append(key_current_node)
        animateNodeDir()
        for goal in goal_nodes:
            if key_current_node == goal:
                reached_goal, cumulative_cost_goal = True, cost_node
                # visited_string = visited_string + '-->' + key_current_node[0]
                console.configure(
                    text='Visited nodes through search are: ' + visited_string + '\nCumulative Cost: ' + str(
                        cumulative_cost_goal))
                return

        # get all successors of key_current_node
        keys_successors = G_sub.getSuccessors(key_current_node)

        if keys_successors:  # checks if contains successors
            # insert all successors of key_current_node in the queue
            for key_successor in keys_successors:
                cumulative_cost = G_sub.getWeightEdge(key_current_node, key_successor) + cost_node
                queue.insert((key_successor, cumulative_cost), cumulative_cost)

    if reached_goal:
        print('\nReached goal! Cost: %s\n' % cumulative_cost_goal)
    else:
        print('\nUnfulfilled goal.\n')


def a_star():
    console.configure(text='A* Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, G_di, goal_nodes, G_sub, heuristic_dic, nodes_index, animatedNodes
    # UCS uses priority queue, priority is the cumulative cost (smaller cost)
    queue = PriorityQueue()
    animatedNodes = []
    visited_string = start_node

    # expands initial node
    # get the keys of all successors of initial node
    keys_successors = G_sub.getSuccessors(start_node)
    animatedNodes.append(start_node)
    animateNode()

    parent_node = start_node

    # adds the keys of successors in priority queue
    for key_successor in keys_successors:
        weight_inner = G_sub.getWeightEdge(start_node, key_successor) + heuristic_dic[key_successor]
        # each item of queue is a tuple (key, cumulative_cost)
        print(key_successor + "weights " + str(weight_inner))
        queue.insert((key_successor, weight_inner), weight_inner)

    reached_goal, cumulative_cost_goal = False, -1
    while not queue.is_empty():
        # remove item of queue, remember: item of queue is a tuple (key, cumulative_cost)
        key_current_node, cost_node = queue.remove()
        # cost_node = cost_node - heuristic_dic[key_current_node]
        visited_nodes.append(key_current_node)
        visited_string = visited_string + '-->' + key_current_node
        animatedNodes.append(key_current_node)
        animateNode()
        for goal in goal_nodes:
            if key_current_node == goal:
                reached_goal, cumulative_cost_goal = True, cost_node
                # visited_string = visited_string + '-->' + key_current_node[0]
                console.configure(text="Visited nodes are : " + visited_string + "\nwith cost " + str(cumulative_cost_goal))
                return

        # get all successors of key_current_node
        keys_successors = G_sub.getSuccessors(key_current_node)

        if keys_successors:  # checks if contains successors
            # insert all successors of key_current_node in the queue
            for key_successor in keys_successors:
                cumulative_cost = G_sub.getWeightEdge(key_current_node, key_successor) + cost_node - heuristic_dic[key_current_node]
                cumulative_cost = cumulative_cost + heuristic_dic[key_successor]
                print(key_successor + "weights " + str(cumulative_cost))
                queue.insert((key_successor, cumulative_cost), cumulative_cost)
                # eVisited.append((key_current_node, key_successor))

    if reached_goal:
        print('\nReached goal!')
    else:
        print('\nUnfulfilled goal.\n')


def a_star_directed():
    console.configure(text='A* Algorithm started for directed graph.')
    global start_node, G, visited_nodes, G_di, goal_nodes, G_sub, heuristic_dic, animatedNodes
    # UCS uses priority queue, priority is the cumulative cost (smaller cost)
    queue = PriorityQueue()
    visited_string = start_node
    animatedNodes = []
    # expands initial node
    # get the keys of all successors of initial node
    keys_successors = G_sub.getSuccessors(start_node)
    animatedNodes.append(start_node)
    animateNodeDir()

    # adds the keys of successors in priority queue
    for key_successor in keys_successors:
        weight_inner = G_sub.getWeightEdge(start_node, key_successor) + heuristic_dic[key_successor]
        # each item of queue is a tuple (key, cumulative_cost)
        queue.insert((key_successor, weight_inner), weight_inner)

    reached_goal, cumulative_cost_goal = False, -1
    while not queue.is_empty():
        # remove item of queue, remember: item of queue is a tuple (key, cumulative_cost)
        key_current_node, cost_node = queue.remove()
        visited_nodes.append(key_current_node)
        visited_string = visited_string + '-->' + key_current_node
        animatedNodes.append(key_current_node)
        animateNodeDir()
        for goal in goal_nodes:
            if key_current_node == goal:
                reached_goal, cumulative_cost_goal = True, cost_node
                # visited_string = visited_string + '-->' + key_current_node[0]
                console.configure(text="Visited nodes are : " + visited_string)
                return

        # get all successors of key_current_node
        keys_successors = G_sub.getSuccessors(key_current_node)

        if keys_successors:  # checks if contains successors
            # insert all successors of key_current_node in the queue
            for key_successor in keys_successors:
                cumulative_cost = G_sub.getWeightEdge(key_current_node, key_successor) + cost_node - heuristic_dic[key_current_node]
                cumulative_cost = cumulative_cost + heuristic_dic[key_successor]
                queue.insert((key_successor, cumulative_cost), cumulative_cost)
                eVisited.append((key_current_node, key_successor))

    if reached_goal:
        print('\nReached goal!')
    else:
        print('\nUnfulfilled goal.\n')


def greedy():
    console.configure(text='Greedy Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, G_di, goal_nodes, verbose, time_sleep, G_sub, heuristic_dic, animatedNodes
    # UCS uses priority queue, priority is the cumulative cost (smaller cost)
    queue = []
    visited_string = start_node
    animatedNodes = []
    # expands initial node
    # get the keys of all successors of initial node
    keys_successors = G_sub.getSuccessors(start_node)
    animatedNodes.append(start_node)
    animateNode()
    # adds the keys of successors in priority queue
    for key_successor in keys_successors:
        inner_heuristic = heuristic_dic[key_successor]
        queue.append((inner_heuristic, key_successor))
        queue.sort(reverse=False)

    reached_goal = False
    while queue:
        # remove item of queue, remember: item of queue is a tuple (key, cumulative_cost)
        h_clone, key_current_node = queue.pop(0)
        visited_string = visited_string + '-->' + key_current_node
        animatedNodes.append(key_current_node)
        animateNode()
        for goal in goal_nodes:
            if key_current_node == goal:
                reached_goal = True
                console.configure(text="Visited nodes are : " + visited_string)
                return

        # get all successors of key_current_node
        keys_successors = G_sub.getSuccessors(key_current_node)

        if keys_successors:  # checks if contains successors
            # insert all successors of key_current_node in the queue
            for key_successor in keys_successors:
                inner_heuristic = heuristic_dic[key_successor]
                queue.append((inner_heuristic, key_successor))
                queue.sort(reverse=False)


def greedy_directed():
    console.configure(text='Greedy Algorithm started for directed graph.')
    global start_node, G, visited_nodes, G_di, goal_nodes, verbose, time_sleep, G_sub, heuristic_dic, animatedNodes
    # UCS uses priority queue, priority is the cumulative cost (smaller cost)
    queue = []
    visited_string = start_node
    animatedNodes = []
    # expands initial node
    # get the keys of all successors of initial node
    keys_successors = G_sub.getSuccessors(start_node)
    animatedNodes.append(start_node)
    animateNodeDir()
    # adds the keys of successors in priority queue
    for key_successor in keys_successors:
        inner_heuristic = heuristic_dic[key_successor]
        queue.append((inner_heuristic, key_successor))
        queue.sort(reverse=False)

    reached_goal = False
    while queue:
        # remove item of queue, remember: item of queue is a tuple (key, cumulative_cost)
        h_clone, key_current_node = queue.pop(0)
        visited_string = visited_string + '-->' + key_current_node
        animatedNodes.append(key_current_node)
        animateNodeDir()
        for goal in goal_nodes:
            if key_current_node == goal:
                reached_goal = True
                console.configure(text="Visited nodes are : " + visited_string)
                return

        # get all successors of key_current_node
        keys_successors = G_sub.getSuccessors(key_current_node)

        if keys_successors:  # checks if contains successors
            # insert all successors of key_current_node in the queue
            for key_successor in keys_successors:
                inner_heuristic = heuristic_dic[key_successor]
                queue.append((inner_heuristic, key_successor))
                queue.sort(reverse=False)

def dfs1(l1,l2,l3,l4,l5):
    console.configure(text='DFS Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, animatedNodes
    visited = {node: False for node in G.nodes}
    visited_string = ''
    queue = [start_node]
    visited[start_node] = True
    result = []
    eVisited = []  # edges which have been considered
    ePath = []
    path = {}
    path[start_node] = start_node
    last_node = start_node
    animatedNodes=[]
    testlist=[]
    l3=g.DFS(start_node,testlist)
    lists=[l1,l1+l2,l3,l1+l2+l3+l4,l1+l2+l3+l4+l5]
    # animatedNodes = l1
    # animateNode()
    # animatedNodes = animatedNodes+l2
    #
    # animateNode()
    # animatedNodes = animatedNodes + l3
    # animateNode()
    for mylist in lists:

        for i in range(0,len(mylist)):
            cur_node = mylist[i]
            result.append(cur_node)
            animatedNodes.append(cur_node)
            # animatedNodes=l1
            animateNode()
            if last_node != start_node:
                eVisited.append((last_node, cur_node))
                print(eVisited)
                animateG(eVisited, ePath)
                last_node = cur_node
            visited_nodes.append(cur_node)
            for goal in goal_nodes:
                if cur_node == goal:
                    visited_string = visited_string + ' , '.join(result)
                    console.configure(text="Visited Nodes are : " + visited_string)
                    console.update()
                    curr_goal = goal
                    actual_path_nodes = [curr_goal]
                    while curr_goal != start_node:
                        curr_goal = path[curr_goal]
                        actual_path_nodes.insert(0, curr_goal)
                    for i in range(len(actual_path_nodes)):
                        if actual_path_nodes[i] != goal_nodes[0]:
                            ePath.append((actual_path_nodes[i], actual_path_nodes[i + 1]))
                            animateG(eVisited, ePath)
                    console.configure(
                        text="Visited Nodes are : " + visited_string + "\nActual Path is " + str(actual_path_nodes))
                    return
            for node in G.neighbors(cur_node):
                if not visited[node]:
                    queue.append(node)
                    path[node] = cur_node
                    visited[node] = True
    visited_string = visited_string + ' , '.join(result)
    console.configure(text="Goal not found  , Visited Nodes are : " + visited_string)




def dfs2(l1,l2,l3,l4,l5):
    console.configure(text='DFS Algorithm started for undirected graph.')
    global start_node, G, visited_nodes, animatedNodes
    visited = {node: False for node in G.nodes}
    visited_string = ''
    queue = [start_node]
    visited[start_node] = True
    result = []
    eVisited = []  # edges which have been considered
    ePath = []
    path = {}
    path[start_node] = start_node
    last_node = start_node
    animatedNodes=[]
    testlist=[]
    l3=g.DFS(start_node,testlist)
    lists=[l1,l1+l2,l3,l1+l2+l3+l4,l1+l2+l3+l4+l5]
    # animatedNodes = l1
    # animateNode()
    # animatedNodes = animatedNodes+l2
    #
    # animateNode()
    # animatedNodes = animatedNodes + l3
    # animateNode()
    for mylist in lists:

        for i in range(0,len(mylist)):
            cur_node = mylist[i]
            result.append(cur_node)
            animatedNodes.append(cur_node)
            # animatedNodes=l1
            animateNode()
            if last_node != start_node:
                eVisited.append((last_node, cur_node))
                print(eVisited)
                animateGDir(eVisited, ePath)
                last_node = cur_node
            visited_nodes.append(cur_node)
            for goal in goal_nodes:
                if cur_node == goal:
                    visited_string = visited_string + ' , '.join(result)
                    console.configure(text="Visited Nodes are : " + visited_string)
                    console.update()
                    curr_goal = goal
                    actual_path_nodes = [curr_goal]
                    while curr_goal != start_node:
                        curr_goal = path[curr_goal]
                        actual_path_nodes.insert(0, curr_goal)
                    for i in range(len(actual_path_nodes)):
                        if actual_path_nodes[i] != goal_nodes[0]:
                            ePath.append((actual_path_nodes[i], actual_path_nodes[i + 1]))
                            animateGDir(eVisited, ePath)
                    console.configure(
                        text="Visited Nodes are : " + visited_string + "\nActual Path is " + str(actual_path_nodes))
                    return
            for node in G_di.successors(cur_node):
                if not visited[node]:
                    queue.append(node)
                    path[node] = cur_node
                    visited[node] = True
        visited_string = visited_string + ' , '.join(result)
        console.configure(text="Goal not found  , Visited Nodes are : " + visited_string)




def ids():
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    global inputList
    for i in range(0, len(inputList), 2):
        if ((inputList[i] in list1 or inputList[i] in list2 or inputList[i] in list3 or inputList[i] in list4 or inputList[i] in list5) and ( inputList[i + 1] in list1 or inputList[i + 1] in list2 or inputList[i + 1] in list3 or inputList[ i + 1] in list4 or inputList[i + 1] in list5)):
            continue
        elif (inputList[i] not in list1 and inputList[i] not in list2 and inputList[i] not in list3 and inputList[i] not in list4 and inputList[i] not in list5):
            list1.append(inputList[i])
            list2.append(inputList[i + 1])
        elif (inputList[i] in list1):
            list2.append(inputList[i + 1])
        elif (inputList[i] in list2):
            list3.append(inputList[i + 1])
        elif (inputList[i] in list3):
            list4.append(inputList[i + 1])
        elif (inputList[i] in list4):
            list5.append(inputList[i + 1])
    print(list1)
    print(list2)
    print(list3)
    print(list4)
    print(list5)
    global g
    g=Graphical()
    for i in range(0,len(inputList),2):
        g.addEdge(inputList[i],inputList[i+1])
    dfs1(list1,list2,list3,list4,list5)


def ids_directed():
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    global inputList
    for i in range(0, len(inputList), 2):
        if ((inputList[i] in list1 or inputList[i] in list2 or inputList[i] in list3 or inputList[i] in list4 or inputList[i] in list5) and ( inputList[i + 1] in list1 or inputList[i + 1] in list2 or inputList[i + 1] in list3 or inputList[ i + 1] in list4 or inputList[i + 1] in list5)):
            continue
        elif (inputList[i] not in list1 and inputList[i] not in list2 and inputList[i] not in list3 and inputList[i] not in list4 and inputList[i] not in list5):
            list1.append(inputList[i])
            list2.append(inputList[i + 1])
        elif (inputList[i] in list1):
            list2.append(inputList[i + 1])
        elif (inputList[i] in list2):
            list3.append(inputList[i + 1])
        elif (inputList[i] in list3):
            list4.append(inputList[i + 1])
        elif (inputList[i] in list4):
            list5.append(inputList[i + 1])
    print(list1)
    print(list2)
    print(list3)
    print(list4)
    print(list5)
    global g
    g=Graphical()
    for i in range(0,len(inputList),2):
        g.addEdge(inputList[i],inputList[i+1])
    dfs2(list1,list2,list3,list4,list5)




def reset():
    global G_di, G, G_sub, fromNodeString, toNodeString, weight, heuristic, node, heuristic_dic, nodes_index, goal_nodes, start_node, color_map, visited_nodes, animatedNodes, console
    G_di = nx.DiGraph()
    G = nx.Graph()
    G_sub = Graph()
    fromNodeString = ""
    toNodeString = ""
    weight = ""
    heuristic = 0
    node = ''
    heuristic_dic.clear()
    nodes_index.clear()
    goal_nodes.clear()
    start_node = ""
    color_map.clear()
    visited_nodes.clear()
    animatedNodes.clear()
    console.configure(text="Reset Completed")


def default():
    global G, G_di, goal_nodes, start_node, color_map
    G.add_weighted_edges_from([('A', 'B', 4), ('A', 'C', 7), ('C', 'D', 1), ('B', 'D', 3)])
    G_di.add_weighted_edges_from([('A', 'B', 4), ('A', 'C', 7), ('C', 'D', 1), ('B', 'D', 3)])
    goal_nodes.append('D')
    start_node = 'A'
    heuristic_dic['A'] = 5
    heuristic_dic['B'] = 1
    heuristic_dic['C'] = 3
    heuristic_dic['D'] = 0
    nodes_index['A'] = 0
    nodes_index['B'] = 1
    nodes_index['C'] = 2
    nodes_index['D'] = 3
    G_sub.addNode('A')
    G_sub.addNode('B')
    G_sub.addNode('C')
    G_sub.addNode('D')
    G_sub.connect('A', 'B', 4)
    G_sub.connect('A', 'C', 7)
    G_sub.connect('C', 'D', 1)
    G_sub.connect('B', 'D', 3)
    color_map = []
    console.configure(text="Default Values Entered")


# GUI starts here
root = Tk()
root.geometry(str(WIDTH) + "x" + str(HEIGHT))
root.title("Artificial Intelligence Final Project")
root.configure(bg='#ffffff')
Label(root, text="From Node", font=("Montserrat", 12), fg='black',bg='white').grid(column=1, row=0, padx=2)
Label(root, text="To Node", font=("Montserrat", 12), fg='black',bg='white').grid(column=2, row=0, padx=2)
Label(root, text="Weight", font=("Montserrat", 12), fg='black',bg='white').grid(column=3, row=0, padx=2)

fromNodeField = Entry(root, width=5, justify="center", bg='white')
fromNodeField.grid(column=1, row=1, padx=2)
toNodeField = Entry(root, width=5, justify="center", bg='white')
toNodeField.grid(column=2, row=1, padx=2)
weightField = Entry(root, width=5, justify="center", bg='white')
weightField.grid(column=3, row=1, padx=2)
addEdgeBtn = Button(root, text="Add Edge",fg='black',bg='yellow', width=18, height=2, command=addEdgeToGraph, font=("Montserrat", 10))
addEdgeBtn.grid(column=5, row=1, columnspan=3)

divider1 = Label(root, text="")

Label(root, text="Node", font=("Montserrat", 12), fg='black',bg='white').grid(column=1, row=2, padx=2)
Label(root, text="Heuristics", font=("Montserrat", 12), fg='black',bg='white').grid(column=2, row=2, padx=2)

NodeField = Entry(root, width=5, justify="center", bg='white')
NodeField.grid(column=1, row=3, padx=2)
HeuristicValueField = Entry(root, width=5, justify="center", bg='white')
HeuristicValueField.grid(column=2, row=3, padx=2)
addHeuristicBtn = Button(root, text="Add Heuristics",fg='black',bg='yellow', width=18, height=2, command=lambda: addHeuristic(), font=("Montserrat", 10))
addHeuristicBtn.grid(column=5, row=3, columnspan=3)

Label(root, text="Start", font=("Montserrat", 12), fg='black', bg='white').grid(column=1, row=4, padx=2)
Label(root, text="Goal/s", font=("Montserrat", 12), fg='black',bg='white').grid(column=2, row=4, padx=2)

startField = Entry(root, width=5, justify="center", bg='white')
startField.grid(column=1, row=5, padx=2)
goalField = Entry(root, width=5, justify="center", bg='white')
goalField.grid(column=2, row=5, padx=2)
addStartAndGoal = Button(root, text="Add Start and Goal/s",fg='black',bg='yellow', width=18, height=3, command=lambda: addGoalNode(), font=("Montserrat", 10)).grid(column=5, row=5, columnspan=3)

photo=PhotoImage(file=r"images/bfsun.png")
photo2=PhotoImage(file=r"images/dfsun.png")
photo3=PhotoImage(file=r"images/ucsun.png")
photo4=PhotoImage(file=r"images/idsun.png")
photo5=PhotoImage(file=r"images/asun.png")
photo6=PhotoImage(file=r"images/gun.png")
photo7=PhotoImage(file=r"images/gd.png")
photo8=PhotoImage(file=r"images/bfsd.png")
photo9=PhotoImage(file=r"images/dfsd.png")
photo10=PhotoImage(file=r"images/ucsd.png")
photo11=PhotoImage(file=r"images/idsd.png")
photo12=PhotoImage(file=r"images/asd.png")




Label(root, text="Directed", font=("Montserrat", 12), fg='black',bg='white').grid(column=1, row=6, padx=2)
Label(root, text="Uniformed", font=("Montserrat", 12), fg='black',bg='white').grid(column=2, row=6, padx=2)
Label(root, text="Directed", font=("Montserrat", 12), fg='black',bg='white').grid(column=5, row=6, padx=2)
Label(root, text="Informed", font=("Montserrat", 12), fg='black',bg='white').grid(column=6, row=6, padx=2)
Button(root, image=photo, command=lambda: bfsDirected(),borderwidth=0).grid(column=1, row=7)
Button(root, image=photo2, command=lambda: dfs_directed(),borderwidth=0).grid(column=2, row=7, padx=2)
Button(root, image=photo3, command=lambda: ucs_directed(),borderwidth=0).grid(column=3, row=7, padx=2)
Button(root, image=photo6,borderwidth=0, command=lambda: greedy_directed()).grid(column=6, row=7, padx=2)
Button(root, image=photo5,borderwidth=0, command=lambda: a_star_directed()).grid(column=5, row=7, padx=2)
Button(root, image=photo4, command=lambda: ids_directed(),borderwidth=0).grid(column=4, row=7, padx=2)
Button(root, text="Draw",fg='white',bg='red', width=7, height=2, command=lambda: drawDiGraph(), font=("Montserrat", 12)).grid(column=7, row=7, padx=2)



Label(root, text="Un-directed", font=("Montserrat", 12), fg='black',bg='white').grid(column=1, row=8, padx=2)
Label(root, text="Uniformed", font=("Montserrat", 12), fg='black',bg='white').grid(column=2, row=8, padx=2)
Label(root, text="Un-directed", font=("Montserrat", 12), fg='black',bg='white').grid(column=5, row=8, padx=2)
Label(root, text="informed", font=("Montserrat", 12), fg='black',bg='white').grid(column=6, row=8, padx=2)
Button(root,image=photo8,borderwidth=0, command=lambda: bfs()).grid(column=1, row=9)
Button(root, image=photo9,borderwidth=0, command=lambda: dfs()).grid(column=2, row=9, padx=2)
Button(root, image=photo10,borderwidth=0, command=lambda: ucs()).grid(column=3, row=9, padx=2)
Button(root, image=photo7,borderwidth=0,command=lambda: greedy()).grid(column=6, row=9, padx=2)
Button(root, image=photo12,borderwidth=0, command=lambda: a_star(), font=("Montserrat", 12)).grid(column=5, row=9, padx=2)
Button(root, image=photo11,borderwidth=0, command=lambda: ids(), font=("Montserrat", 12)).grid(column=4, row=9, padx=2)
Button(root, text="Draw",fg='white',bg='blue', width=7, height=2, command=lambda: drawGraph(), font=("Montserrat", 12)).grid(column=7, row=9, padx=2)

Button(root, text="Reset",fg='black',bg='yellow', width=30, height=2, command=lambda: reset(), font=("Montserrat", 12)).grid(column=3, row=10, padx=2, columnspan=4)
Button(root, text="Default",fg='black',bg='yellow', width=15, height=2, command=lambda: default(), font=("Montserrat", 12)).grid(column=1, row=10, padx=2, columnspan=2)

f = Figure(figsize=(7, 5), dpi=100)
a = f.add_subplot(111)

canvas = FigureCanvasTkAgg(f, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)

console = Label(root, text="Console data here", font=("Montserrat", 14), fg='black', bg='white', width=50, height=2)
console.grid(column=0, row=10, padx=2)

root.mainloop()
