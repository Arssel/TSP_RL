import numpy as np
from src.utils import path_distance_new as path_distance
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def nearest_neighbour(distances):
    bsz, n, _ = distances.shape
    routes = np.zeros((bsz, n), dtype=int)
    for bid in np.arange(bsz):
        added_nodes = []
        last_point = np.random.randint(0, n)
        added_nodes.append(last_point)
        route = np.array([last_point], dtype=int)
        sequence = []
        sequence.append(route)
        n_range = np.arange(n)
        not_visited = n_range[n_range != last_point]
        for i in range(1,n):
            new_point = np.abs(distances[bid, last_point, :] - distances[bid, last_point, not_visited].min()).argmin()
            route = np.concatenate((route, [new_point]))
            not_visited = not_visited[not_visited != new_point]
            last_point = new_point
            added_nodes.append(last_point)
            sequence.append(route)
        routes[bid, :] = route
    return path_distance(distances, routes), routes, sequence, added_nodes

def insert_heuristic(distances, insert_type='remote'):
    bsz, n, _ = distances.shape
    routes = np.zeros((bsz, n), dtype=int)
    n_range = np.arange(n)
    
    for bid in np.arange(bsz):
        added_nodes = []
        first_point = np.random.randint(0, n)
        added_nodes.append(first_point)
        not_visited = n_range[n_range != first_point]
        if insert_type == 'remote':
            second_point = np.abs(distances[bid, first_point, :] - distances[bid, first_point, not_visited].max()).argmin()
        else:
            second_point = np.abs(distances[bid, first_point, :] - distances[bid, first_point, not_visited].min()).argmin()
        not_visited = not_visited[not_visited != second_point]
        route = np.array([first_point, second_point], dtype=int)
        added_nodes.append(second_point)
        sequence = []
        sequence.append(np.append(route, route[0]))
        for i in range(2,n):
            if insert_type == 'remote':
                node = remotest_node(route, not_visited, distances[bid])
            else:
                node = closest_node(route, not_visited, distances[bid])
            added_nodes.append(node)
            route = find_place_for_node(route, node, distances[bid])
            sequence.append(np.append(route, route[0]))
            not_visited = not_visited[not_visited != node]
        routes[bid, :] = route
    return path_distance(distances, routes), routes, sequence, added_nodes

def closest_node(route, not_visited, distances):
    min_index = -1
    min_distance = 2
    for i in np.arange(len(route)):
        i_dist = np.abs(distances[route[i], :] - distances[route[i], not_visited].min()).argmin()
        if distances[route[i], i_dist] < min_distance:
            min_index = i_dist
            min_distance = distances[route[i], i_dist]
    return min_index

def remotest_node(route, not_visited, distances):
    max_index = -1
    max_distance = 0
    for i in not_visited:
        i_dist = distances[i, route].min()
        if i_dist > max_distance:
            max_index = i
            max_distance = i_dist
    return max_index

def find_place_for_node(route, node, distances):
    min_distance = distances.shape[1]
    min_index = -1
    for i in np.arange(len(route)):
        temp_route = np.insert(route, i, node)
        length = path_distance(distances[None, :, :], temp_route.reshape(-1, temp_route.size))
        if length < min_distance:
            min_index = i
            min_distance = length
    return np.insert(route, min_index, node)    



def total_distance(manager, routing, solution):
    """
    Compute route distance
    """
    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    return route_distance


def compute_distance(distance_matrix, eps=1e-5, time_limit=1):

    data = {}
    data['distance_matrix'] = distance_matrix/eps
    data['num_vehicles'] = 1
    data['depot'] = 0

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """
        Returns the distance between the two nodes.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]    

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = time_limit
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    return total_distance(manager, routing, solution)*eps