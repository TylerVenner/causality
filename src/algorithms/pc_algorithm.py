import pandas as pd
import networkx as nx
import pingouin as pg
from itertools import combinations, permutations
from typing import Dict, Set, Tuple

def _get_neighbors(graph: nx.Graph, node) -> Set:
    """Helper to get the set of current neighbors for a node."""
    return set(graph.neighbors(node))

def partial_correlation_test(data: pd.DataFrame, i: str, j: str, S: Set[str], alpha: float) -> bool:
    """
    Performs a conditional independence test using partial correlation.
    
    Args:
        data: The DataFrame.
        i, j: The two variables to test.
        S: The conditioning set.
        alpha: The significance level.

    Returns:
        bool: True if i and j are independent given S (p-value > alpha), False otherwise.
    """
    # Handle the edge case where n_samples is too small for the test
    n_samples = len(data)
    if n_samples < len(S) + 3:
        # Not enough data to test; conservatively assume dependence
        return False

    # Use pingouin to calculate partial correlation and p-value
    # It correctly handles S=None or S=[] for order-0 correlation
    try:
        result = pg.partial_corr(data=data, x=i, y=j, covar=list(S) if S else None)
        p_value = result['p-val'].iloc[0]
        
        # We cannot reject H0 (independence)
        return p_value > alpha
    except Exception as e:
        # Handle potential numerical errors (e.g., singular matrix)
        # Conservatively assume dependence
        print(f"CI Test Error ({i}, {j} | {S}): {e}")
        return False


def pc_step_1_skeleton(data: pd.DataFrame, alpha: float) -> Tuple[nx.Graph, Dict]:
    """
    Executes Step 1 of the PC algorithm to find the graph skeleton.
    
    Args:
        data: The observational data.
        alpha: The significance level for CI tests.
        
    Returns:
        Tuple containing:
        - nx.Graph: The undirected graph skeleton.
        - dict: The sepset dictionary.
    """
    nodes = list(data.columns)
    skeleton = nx.complete_graph(nodes)
    sepset = {}
    
    k = 0
    while True:
        k_changed = False
        edges_to_remove = []
        
        for (i, j) in skeleton.edges():
            # Correction: Use CURRENT neighbors (adj) for subset generation
            adj_i = _get_neighbors(skeleton, i) - {j}
            adj_j = _get_neighbors(skeleton, j) - {i}
            
            # Check neighbors of i
            if len(adj_i) >= k:
                for S in combinations(adj_i, k):
                    if partial_correlation_test(data, i, j, set(S), alpha):
                        edges_to_remove.append((i, j))
                        sepset[(i, j)] = set(S)
                        sepset[(j, i)] = set(S) # Store symmetric separating set
                        k_changed = True
                        break # Move to next edge
                if (i, j) in edges_to_remove:
                    continue # Already found separating set for this edge
            
            # Check neighbors of j (if not found for i)
            if len(adj_j) >= k:
                for S in combinations(adj_j, k):
                    if partial_correlation_test(data, i, j, set(S), alpha):
                        edges_to_remove.append((i, j))
                        sepset[(i, j)] = set(S)
                        sepset[(j, i)] = set(S)
                        k_changed = True
                        break # Move to next edge
        
        skeleton.remove_edges_from(edges_to_remove)
        
        if not k_changed:
            break # No more edges were removed at this k, so we're done
        
        k += 1
        
    return skeleton, sepset

def pc_step_1_skeleton_with_logging(data: pd.DataFrame, alpha: float) -> tuple[nx.Graph, dict, list[str]]:
    """
    Executes Step 1 of the PC algorithm to find the graph skeleton.
    Returns the skeleton, sepset dictionary, and a debug log.
    
    This is a standalone version with logging for the PC Algorithm Demo page.
    
    Args:
        data: The observational data (DataFrame with variables as columns)
        alpha: The significance level for CI tests (typically 0.05)
        
    Returns:
        Tuple containing:
        - nx.Graph: The undirected graph skeleton
        - Dict: The sepset dictionary {(i,j): Set of conditioning variables}
        - List[str]: Debug log of all tests performed
    """
    nodes = list(data.columns)
    skeleton = nx.complete_graph(nodes)
    sepset = {}
    log = []
    
    k = 0
    while True:
        k_changed = False
        log.append(f"--- Testing with conditioning set size k = {k} ---")
        edges_to_remove = []
        
        # Iterate over a copy of edges, as we modify the graph
        for (i, j) in list(skeleton.edges()):
            adj_i = _get_neighbors(skeleton, i) - {j}
            adj_j = _get_neighbors(skeleton, j) - {i}
            
            found_separator = False
            
            # Use the smaller adjacency set for efficiency
            adj_set = adj_i if len(adj_i) <= len(adj_j) else adj_j
            
            if len(adj_set) >= k:
                for S in combinations(adj_set, k):
                    if partial_correlation_test(data, i, j, set(S), alpha, log):
                        edges_to_remove.append((i, j))
                        sepset[(i, j)] = set(S)
                        sepset[(j, i)] = set(S)
                        k_changed = True
                        found_separator = True
                        log.append(f"REMOVING edge {i} -- {j} based on S = {S}")
                        break
            if found_separator:
                continue
        
        skeleton.remove_edges_from(edges_to_remove)
            
        k += 1
        
        # Check if we can even form a conditioning set of size k
        if all(len(_get_neighbors(skeleton, n)) < k for n in nodes):
            log.append(f"Stopping: No node has {k} neighbors left.")
            break
            
    log.append("--- Skeleton search complete ---")
    return skeleton, sepset, log

def pc_step_2_orient_colliders(skeleton: nx.Graph, sepset: Dict) -> nx.DiGraph:
    """
    Executes Step 2 of the PC algorithm to orient v-structures (colliders).
    
    Args:
        skeleton: The undirected skeleton from Step 1.
        sepset: The separating set dictionary from Step 1.
        
    Returns:
        nx.DiGraph: A partially directed graph (PDAG) containing oriented colliders.
    """
    pdag = nx.DiGraph(skeleton) # Start with all edges as bi-directional
    
    for k in skeleton.nodes():
        # Find all uncoupled pairs (i, j) that are neighbors of k
        neighbors_k = _get_neighbors(skeleton, k)
        for (i, j) in combinations(neighbors_k, 2):
            # Check if i and j are uncoupled (not adjacent)
            if not skeleton.has_edge(i, j):
                # This is a v-structure i - k - j
                # Now check if k is in the separating set of (i, j)
                
                # Check sepset. Use get() to handle cases where (i,j) had no sepset (which shouldn't happen if not adj)
                sep = sepset.get((i, j)) 
                
                if sep is None or k not in sep:
                    # k is NOT in sepset(i, j), so orient as collider i -> k <- j
                    # Remove the k -> i and k -> j edges from the DiGraph
                    if pdag.has_edge(k, i): pdag.remove_edge(k, i)
                    if pdag.has_edge(k, j): pdag.remove_edge(k, j)
                    
    return pdag


def pc_step_3_orient_remaining(pdag: nx.DiGraph) -> nx.DiGraph:
    """
    Executes Step 3 of the PC algorithm, applying Meek's 4 orientation rules
    iteratively until no more edges can be oriented.
    
    Meek's Rules (from Meek 1995):
    R1: Orient i-j into i->j if k->i and k,j not adjacent
    R2: Orient i-j into i->j if i->k->j
    R3: Orient i-j into i->j if i-k->j, i-l->j, and k,l not adjacent
    R4: Orient i-j into i->j if i-k->l, k->l->j, and k,j not adjacent
    
    Args:
        pdag: The partially directed graph from Step 2.
        
    Returns:
        nx.DiGraph: The final CPDAG.
    """
    
    def _has_undirected_edge(G, u, v):
        """Check if u-v is undirected (both u->v and v->u exist)"""
        return G.has_edge(u, v) and G.has_edge(v, u)

    def _has_directed_edge(G, u, v):
        """Check if u->v is directed (u->v exists but v->u does not)"""
        return G.has_edge(u, v) and not G.has_edge(v, u)

    def _is_adjacent(G, u, v):
        """Check if u and v are adjacent (any edge between them)"""
        return G.has_edge(u, v) or G.has_edge(v, u)

    while True:
        made_change = False
        
        # Rule R1: Orient i-j into i->j if there is k->i and k,j not adjacent
        # Pattern: k -> i - j with k and j nonadjacent
        # Reason: Avoid creating v-structure k -> i <- j
        for (i, j) in list(pdag.edges()):
            if not _has_undirected_edge(pdag, i, j):
                continue
            
            # Find k such that k -> i
            for k in pdag.nodes():
                if k == i or k == j:
                    continue
                if _has_directed_edge(pdag, k, i) and not _is_adjacent(pdag, k, j):
                    # Orient i -> j
                    pdag.remove_edge(j, i)
                    made_change = True
                    break
            if made_change:
                break
        if made_change:
            continue

        # Rule R2: Orient i-j into i->j if there is a chain i->k->j
        # Pattern: i -> k -> j with i-j undirected
        # Reason: Avoid creating a cycle
        for (i, j) in list(pdag.edges()):
            if not _has_undirected_edge(pdag, i, j):
                continue
            
            # Find k such that i -> k -> j
            for k in pdag.nodes():
                if k == i or k == j:
                    continue
                if _has_directed_edge(pdag, i, k) and _has_directed_edge(pdag, k, j):
                    # Orient i -> j
                    pdag.remove_edge(j, i)
                    made_change = True
                    break
            if made_change:
                break
        if made_change:
            continue

        # Rule R3: Orient i-j into i->j if there are two chains i-k->j and i-l->j
        # where k and l are nonadjacent
        # Pattern: i - k -> j and i - l -> j with k,l nonadjacent
        # Reason: Avoid creating v-structure k -> j <- l
        for (i, j) in list(pdag.edges()):
            if not _has_undirected_edge(pdag, i, j):
                continue
            
            # Find all k such that i - k -> j
            candidates = []
            for k in pdag.nodes():
                if k == i or k == j:
                    continue
                if _has_undirected_edge(pdag, i, k) and _has_directed_edge(pdag, k, j):
                    candidates.append(k)
            
            # Check if any two candidates are nonadjacent
            if len(candidates) >= 2:
                for k, l in combinations(candidates, 2):
                    if not _is_adjacent(pdag, k, l):
                        # Orient i -> j
                        pdag.remove_edge(j, i)
                        made_change = True
                        break
                if made_change:
                    break
        if made_change:
            continue

        # Rule R4: Orient i-j into i->j if there are chains i-k->l and k->l->j
        # where k and j are nonadjacent
        # Pattern: discriminating path i - k -> l -> j with k,j nonadjacent
        # Reason: Complex case for discriminating paths
        for (i, j) in list(pdag.edges()):
            if not _has_undirected_edge(pdag, i, j):
                continue
            
            # Find k,l such that i - k -> l -> j with k,j nonadjacent
            for k in pdag.nodes():
                if k == i or k == j:
                    continue
                if not _has_undirected_edge(pdag, i, k):
                    continue
                if _is_adjacent(pdag, k, j):
                    continue  # k and j must be nonadjacent
                
                # Now find l such that k -> l -> j
                for l in pdag.nodes():
                    if l == i or l == j or l == k:
                        continue
                    if _has_directed_edge(pdag, k, l) and _has_directed_edge(pdag, l, j):
                        # Orient i -> j
                        pdag.remove_edge(j, i)
                        made_change = True
                        break
                if made_change:
                    break
            if made_change:
                break
        if made_change:
            continue

        # If no changes were made in this full pass, we are done
        if not made_change:
            break
            
    return pdag
