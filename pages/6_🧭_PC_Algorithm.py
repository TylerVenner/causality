import streamlit as st
import pandas as pd
import networkx as nx
import pingouin as pg
import graphviz
from itertools import combinations, permutations
from typing import List, Dict, Set, Tuple

import src.simulations.pc_simulation as sim
import src.plotting.charts as charts
from src.algorithms.pc_algorithm import pc_step_2_orient_colliders, pc_step_3_orient_remaining

import src.simulations.independence_sim as sim_indep

st.set_page_config(
    page_title="PC Algorithm",
    page_icon="🧭",
    layout="wide"
)


def _get_neighbors(graph: nx.Graph, node) -> Set:
    """Helper to get the set of current neighbors for a node."""
    return set(graph.neighbors(node))

def partial_correlation_test(data: pd.DataFrame, i: str, j: str, S: Set[str], alpha: float, log: List[str]) -> bool:
    """
    Performs a conditional independence test using partial correlation.
    Returns True if independent (p > alpha), False otherwise.
    """
    n_samples = len(data)
    if n_samples < len(S) + 3:
        log.append(f"SKIPPED: {i} _||_ {j} | {S} (n_samples={n_samples} is too small for |S|={len(S)})")
        return False # Conservatively assume dependence

    try:
        result = pg.partial_corr(data=data, x=i, y=j, covar=list(S) if S else None)
        p_value = result['p-val'].iloc[0]
        
        is_independent = p_value > alpha
        
        verdict = "INDEPENDENT" if is_independent else "Dependent"
        log.append(f"Test: {i} _||_ {j} | {S}?  p-val: {p_value:.4f} > {alpha}.  Verdict: {verdict}")
        
        return is_independent
    except Exception as e:
        log.append(f"ERROR: {i} _||_ {j} | {S}. Error: {e}")
        return False # Conservatively assume dependence

def pc_step_1_skeleton_with_logging(data: pd.DataFrame, alpha: float) -> Tuple[nx.Graph, Dict, List[str]]:
    """
    Executes Step 1 of the PC algorithm to find the graph skeleton.
    Returns the skeleton, sepset dictionary, and a debug log.
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

st.title("🧭 The PC Algorithm")
st.markdown(
    """
    We'll use the principles from the previous page, the Causal Markov Property and Faithfulness, to discover a causal graph from data.
    """
)

st.header("Part 1: How the PC Algorithm Works")
st.markdown(
    """
    The **PC Algorithm** (named after its creators, Peter Spirtes and Clark Glymour) is the classic **constraint-based** algorithm for causal discovery. It assumes the Causal Markov and Faithfulness conditions hold, meaning the graph structure and conditional independencies in the data are a perfect match.
    
    The algorithm works in three main phases, just like a detective:
    1.  Find all conditional independencies.
    2.  Orient colliders.
    3.  Deduce the rest (propagate orientations).
    """
)

with st.container(border=True):
    st.subheader("Phase 1: The Adjacency Search (Find the Skeleton)")
    st.markdown(
        """
        It starts by assuming a **fully connected undirected graph** (all variables are connected). It then tries to *disprove* connections.

        It iteratively searches for evidence to remove edges by testing conditional independencies, starting with the simplest tests:
        -   **$k=0$:** For every pair of nodes ($X$, $Y$), it tests for marginal independence: $X \perp\kern-5pt\perp Y$? If "yes", the edge $X-Y$ is removed.
        -   **$k=1$:** For all remaining edges $X-Y$, it tests for independence conditioned on *one* neighbor $Z$: $X \perp\kern-5pt\perp Y \mid Z$? If "yes" for any $Z$, the edge $X-Y$ is removed.
        -   **$k=2$:** For remaining edges, it tests conditioning on all *pairs* of neighbors...
        -   ...and so on, until no more edges can be removed.
        
        The result is an undirected graph called the **skeleton**, which represents all adjacencies that could *not* be disproven. The algorithm also saves the "separating set" sepset($X$, $Y$) that made $X$ and $Y$ independent.
        """
    )

with st.container(border=True):
    st.subheader("Phase 2: Orient Colliders (Find v-structures)")
    st.markdown(
        """
        This is the most critical phase for orienting arrows. The algorithm uses the "Collider" rule from the previous page (Berkson's Paradox).
        
        It searches for all "uncoupled triples" (or "v-structures") in the skeleton: $X - Z - Y$, where $X$ and $Y$ are *not* connected.
        
        For each triple, it checks the separating set sepset($X$, $Y$) that was saved from Phase 1.
        
        -   **If $Z$ is NOT in sepset($X$, $Y$):** This is the smoking gun! $X$ and $Y$ were independent *without* conditioning on $Z$. This means $Z$ *must* be a collider. The algorithm orients the arrows: **$X \\to Z \leftarrow Y$**.
        -   **If $Z$ IS in sepset($X$, $Y$):** This means $Z$ was the variable that *created* the independence (like in a Chain or Fork). The algorithm leaves the edges as-is: $X - Z - Y$.
        """
    )

with st.container(border=True):
    st.subheader("Phase 3: Propagate Orientations (Apply Meek's Rules)")
    st.markdown(
        """
        After finding all the colliders, the algorithm applies **Meek's 4 orientation rules** to orient as many remaining undirected edges as possible. These rules are applied iteratively until no more edges can be oriented.
        
        **Rule R1 (Avoid New V-Structures):** If we have $k \\to i - j$ (where $k$ and $j$ are not adjacent), orient as $i \\to j$. 
        - **Why?** Orienting as $i \leftarrow j$ would create a new collider $k \\to i \leftarrow j$ that we would have already found in Phase 2.
        
        **Rule R2 (Avoid Cycles):** If we have $i \\to k \\to j$ and also $i - j$, orient as $i \\to j$.
        - **Why?** Orienting as $i \leftarrow j$ would create a cycle $i \\to k \\to j \\to i$, which is forbidden in a DAG.
        
        **Rule R3 (Avoid New V-Structures, variant):** If we have $i - k \\to j$ and $i - l \\to j$ (where $k$ and $l$ are not adjacent), orient as $i \\to j$.
        - **Why?** Orienting as $i \leftarrow j$ would create a new v-structure $k \\to j \leftarrow l$.
        
        **Rule R4 (Discriminating Paths):** If we have $i - k \\to l \\to j$ (where $k$ and $j$ are not adjacent), orient as $i \\to j$.
        - **Why?** This handles more complex path structures to ensure consistency.
        
        The final output is the **CPDAG (Completed Partially Directed Acyclic Graph)**, which represents the Markov equivalence class—all causal structures consistent with the observed conditional independencies.
        """
    )

st.divider()

st.header("Interactive Demonstration")
st.markdown("Let's see the algorithm in action on a 4-variable \"Diamond\" graph.")

st.sidebar.header("PC Algorithm Controls")
n_samples = 2000
alpha = 0.05
debug_mode = st.sidebar.checkbox("Show CI Test Log (Debug Mode)", value=False)

# Use session state to store the results
if 'pc_results' not in st.session_state:
    st.session_state.pc_results = None

if st.button("Run PC Algorithm", type="primary", use_container_width=True):
    with st.spinner("Generating data and running algorithm..."):
        # 1. Generate data
        data = sim.generate_diamond_data(n_samples)
        
        # 2. Run Step 1
        skeleton, sepset, log = pc_step_1_skeleton_with_logging(data, alpha)
        
        # 3. Run Step 2
        pdag1 = pc_step_2_orient_colliders(skeleton, sepset)
        
        # 4. Run Step 3
        pdag2 = pc_step_3_orient_remaining(pdag1.copy()) # Use a copy
        
        # Store results
        st.session_state.pc_results = {
            "data": data,
            "skeleton": skeleton,
            "sepset": sepset,
            "pdag1": pdag1,
            "pdag2": pdag2,
            "log": log,
            "truth": sim.get_ground_truth_graph()
        }

# Display results if they exist
if st.session_state.pc_results:
    res = st.session_state.pc_results
    
    tab_truth, tab1, tab2, tab3, tab_log = st.tabs([
        "**1. Ground Truth**", 
        "**2. Step 1: Skeleton**", 
        "**3. Step 2: Colliders**", 
        "**4. Step 3: Final Graph (CPDAG)**",
        "**CI Test Log**"
    ])
    
    with tab_truth:
        st.subheader("Ground Truth Graph")
        st.markdown("This is the 'Diamond' SCM we used to generate the data.")
        st.graphviz_chart(res["truth"])
        with st.expander("Show SCM"):
            st.latex(r'''
                \begin{aligned}
                    A &:= N_A \\
                    B &:= 1.0 \cdot A + N_B \\
                    C &:= -1.5 \cdot A + N_C \\
                    D &:= 2.0 \cdot B - 1.0 \cdot C + N_D
                \end{aligned}
            ''')

    with tab1:
        st.subheader("Step 1: Find the Skeleton")
        st.markdown(
            """
            The algorithm starts with a **fully connected undirected graph** (every variable connected to every other) 
            and systematically removes edges by testing conditional independence.
            
            It checks $i \perp\\kern-5pt\\perp j \mid S$ for conditioning sets $S$ of increasing size ($k=0, 1, 2...$).
            When a test returns p-value > $\\alpha$, we conclude independence and remove the edge.
            """
        )

        dot_skeleton = charts.graphviz_from_nx(res["skeleton"], "Discovered Skeleton")
        st.graphviz_chart(dot_skeleton)

        st.markdown("**Edges Removed:** The algorithm found and removed **A — D** and **B — C**.")

        with st.expander("🔬 How Were These Edges Identified and Removed?"):
            st.markdown(
                """
                **At $k=0$ (unconditional tests):**
                All pairs are dependent without conditioning. No edges removed yet.
                
                **At $k=1$ (conditioning on one variable):**
                
                **Removing B — C:**
                - **Test:** $B \perp\\kern-5pt\\perp C \mid \\{A\\}$?
                - **Result:** p-value > 0.05 → **INDEPENDENT**
                - **Action:** Remove edge B—C, save sepset($B$, $C$) = $\\{A\\}$
                - **Why?** $A$ is the common cause (fork: $B \\leftarrow A \\to C$). Conditioning on $A$ blocks 
                the path and makes $B$ and $C$ independent.
                
                **At $k=2$ (conditioning on two variables):**
                
                **Removing A — D:**
                - **Test:** $A \perp\\kern-5pt\\perp D \mid \\{B, C\\}$?
                - **Result:** p-value = 0.52 > 0.05 → **INDEPENDENT** ✓
                - **Action:** Remove edge A—D, save sepset($A$, $D$) = $\\{B, C\\}$
                - **Why?** All paths from $A$ to $D$ go through $B$ or $C$:
                - Path $A \\to B \\to D$ (blocked when conditioning on $B$)
                - Path $A \\to C \\to D$ (blocked when conditioning on $C$)
                
                **Why other edges remained:**
                - **A—B, A—C, B—D, C—D:** No conditioning set made them independent.
                
                **Final skeleton:** $A-B$, $A-C$, $B-D$, $C-D$ (4 edges remain)
                """
            )

    with tab2:
        st.subheader("Step 2: Orient Colliders (v-structures)")
        st.markdown(
            """
            The algorithm now looks for uncoupled triples like $i - k - j$.
            It orients them as a collider $i \\to k \leftarrow j$ **if and only if** the middle node $k$ was **NOT** in the separating set sepset($i$, $j$).
            """
        )
        dot_pdag1 = charts.graphviz_from_nx(res["pdag1"], "Oriented Colliders")
        st.graphviz_chart(dot_pdag1)

        st.info("Undirected edges from the skeleton are shown in gray. The new collider edges $B \\to D$ and $C \\to D$ are shown in black.")

        with st.expander("🎯 Why Does This Orientation Rule Work?"):
            st.markdown(
                """
                **Testing the triple $B - D - C$:**
                
                1. **Are $B$ and $C$ adjacent?** No (we removed B—C in Step 1) 
                2. **What is sepset($B$, $C$)?** $\\{A\\}$ (saved from Step 1)
                3. **Is $D$ in sepset($B$, $C$)?** No ($D \\notin \\{A\\}$) 
                4. **Conclusion:** Orient as collider $B \\to D \\leftarrow C$
                
                **Why does this logic work?**
                
                For the triple $B - D - C$, there are only three possible causal structures:
                
                1. **Chain:** $B \\to D \\to C$ (or reversed)
                - $D$ is on the path from $B$ to $C$
                - To make $B \perp\\kern-5pt\\perp C$, we must condition on $D$ (block the path)
                - So sepset($B$, $C$) should contain $D$ 
                
                2. **Fork:** $B \\leftarrow D \\to C$
                - $D$ is a common cause of $B$ and $C$
                - To make $B \perp\\kern-5pt\\perp C$, we must condition on $D$ (close backdoor)
                - So sepset($B$, $C$) should contain $D$ 
                
                3. **Collider:** $B \\to D \\leftarrow C$
                - $B$ and $C$ are not confounded (no common cause except $A$)
                - They are independent WITHOUT conditioning on $D$
                - So sepset($B$, $C$) should NOT contain $D$ 
                
                Since sepset($B$, $C$) = $\\{A\\}$ does NOT contain $D$, the only possibility is the collider!
                
                **Why are there no other v-structures in this graph?**
                
                - **$A - B - D$:** $A$ and $D$ not adjacent, but sepset($A$, $D$) = $\\{B, C\\}$. 
                Is $B \\in \\{B, C\\}$? Yes → $B$ IS in the separating set → NOT a collider
                
                - **$A - C - D$:** $A$ and $D$ not adjacent, but sepset($A$, $D$) = $\\{B, C\\}$. 
                Is $C \\in \\{B, C\\}$? Yes → $C$ IS in the separating set → NOT a collider
                
                The only v-structure is $B \\to D \\leftarrow C$.
                """
            )

    with tab3:
        st.subheader("Step 3: Apply Orientation Rules (Meek's Rules)")
        st.markdown(
            """
            The algorithm now applies **Meek's 4 orientation rules** iteratively to orient as many remaining edges as possible.
            These rules use logic to deduce orientations without creating new v-structures or cycles.
            """
        )

        dot_pdag2 = charts.graphviz_from_nx(res["pdag2"], "Final CPDAG")
        st.graphviz_chart(dot_pdag2)

        with st.expander("🔍 Why Don't the Meek Rules Fire Here?"):
            st.markdown(
                """
                After Step 2, we have: $A - B$, $A - C$, $B \\to D$, $C \\to D$. Let's check each rule:
                
                **Rule R1:** Orient $i - j$ as $i \\to j$ if $\\exists k \\to i$ where $k$ and $j$ not adjacent.
                - Pattern: $k \\to i - j$ with $k, j$ nonadjacent
                - For $A - B$: Need $k \\to A$, but no incoming arrows to $A$ exist 
                - For $A - C$: Need $k \\to A$, but no incoming arrows to $A$ exist 
                - **Rule R1 cannot fire.**
                
                **Rule R2:** Orient $i - j$ as $i \\to j$ if $\\exists$ chain $i \\to k \\to j$.
                - Pattern: $i \\to k \\to j$ with $i - j$ undirected
                - For $A - B$: Need $A \\to k \\to B$, but no such directed chain exists 
                - For $A - C$: Need $A \\to k \\to C$, but no such directed chain exists 
                - **Rule R2 cannot fire.**
                
                **Rule R3:** Orient $i - j$ as $i \\to j$ if $\\exists$ two paths $i - k \\to j$ and $i - l \\to j$ where $k, l$ not adjacent.
                - Pattern: Two different intermediate nodes both pointing to $j$
                - For $A - B$: Need nodes pointing to $B$, but nothing points to $B$ 
                - For $A - C$: Need nodes pointing to $C$, but nothing points to $C$ 
                - **Rule R3 cannot fire.**
                
                **Rule R4:** Orient $i - j$ as $i \\to j$ if $\\exists$ path $i - k \\to l \\to j$ where $k, j$ not adjacent.
                - Pattern: Discriminating path structure
                - This complex pattern doesn't exist in our small graph 
                - **Rule R4 cannot fire.**
                
                **Conclusion:** None of Meek's rules can orient $A - B$ or $A - C$. They remain undirected.
                """
            )

        st.markdown("---")

        st.markdown(
            """
            ### Why A—B and A—C Remain Undirected: The Markov Equivalence Class
            
            The edges $A - B$ and $A - C$ remain undirected because they belong to the graph's **Markov equivalence class**—
            a set of different DAGs that all produce identical conditional independencies.
            """
        )

        with st.expander("📊 The Four Equivalent DAGs"):
            st.markdown(
                """
                All of these graphs satisfy **both** independencies we discovered:
                - $B \perp\\kern-5pt\\perp C \mid \\{A\\}$
                - $A \perp\\kern-5pt\\perp D \mid \\{B, C\\}$
                
                | Graph Structure | Satisfies $B \perp\\kern-5pt\\perp C \mid A$? | Satisfies $A \perp\\kern-5pt\\perp D \mid B,C$? |
                |-----------------|----------------------------------------------|------------------------------------------------|
                | $A \\to B \\to D \\leftarrow C \\leftarrow A$ (Ground Truth) | ✓ | ✓ |
                | $A \\leftarrow B \\to D \\leftarrow C \\to A$ | ✓ | ✓ |
                | $A \\to B \\to D \\leftarrow C \\to A$ | ✓ | ✓ |
                | $A \\leftarrow B \\to D \\leftarrow C \\leftarrow A$ | ✓ | ✓ |
                
                **Let's verify this for two examples:**
                
                **Example 1: $A \\to B \\to D \\leftarrow C \\leftarrow A$ (Ground Truth)**
                - $B \perp\\kern-5pt\\perp C \mid A$? Path $B \\leftarrow A \\to C$ blocked at $A$ ✓
                - $A \perp\\kern-5pt\\perp D \mid B,C$? 
                - Path $A \\to B \\to D$ blocked at $B$ ✓
                - Path $A \\to C \\to D$ blocked at $C$ ✓
                
                **Example 2: $A \\leftarrow B \\to D \\leftarrow C \\to A$ (Different orientations!)**
                - $B \perp\\kern-5pt\\perp C \mid A$? Path $B \\to A \\leftarrow C$ blocked at $A$ ✓
                - $A \perp\\kern-5pt\\perp D \mid B,C$?
                - Path $A \\leftarrow B \\to D$ blocked at $B$ ✓
                - Path $A \\leftarrow C \\to D$ blocked at $C$ ✓
                
                **Both produce the same d-separations!** The observational data cannot distinguish between them.
                """
            )

        st.success(
            """
            **Key Insight:** The gray undirected edges ($A - B$, $A - C$) represent **epistemic uncertainty**—
            we know these variables are causally connected, but we cannot determine the direction from 
            observational data alone.
            
            The black directed edges ($B \\to D$, $C \\to D$) form a **v-structure** that is identifiable 
            because $D$ was NOT in sepset($B$, $C$). This structure appears in all four equivalent DAGs.
            """
        )

        st.warning(
            """
            💡 **The Fundamental Limit of Causal Discovery**
            
            This demonstrates a profound limitation: **some causal relationships cannot be determined from 
            observational data alone**, even with:
            - Perfect statistical tests
            - Infinite sample size
            - No confounders or selection bias
            
            To resolve this ambiguity, we would need:
            - **Interventional experiments** (e.g., randomize $A$ and observe effects on $B$ and $C$)
            - **Temporal information** (e.g., if we know $A$ occurs before $B$ and $C$)
            - **Domain knowledge** (e.g., expert knowledge that $A$ causes $B$, not vice versa)
            """
        )
    with tab_log:
        st.subheader("Conditional Independence Test Log")
        if debug_mode:
            st.code("\n".join(res["log"]), language="text")
        else:
            st.info("Enable 'Show CI Test Log (Debug Mode)' in the sidebar to see the detailed log.")

else:
    st.info("Click the 'Run PC Algorithm' button to generate data and discover the causal graph.")

st.divider()
st.header("Part 2: Resolving Ambiguity with Interventions")
st.markdown(
    """
    The PC algorithm correctly told us that the directions of $A - B$ and $A - C$ are **unidentifiable** from observational data.
    
    So, how could we ever find the true ground truth of $A \\to B$ and $A \\to C$?
    
    We need a stronger tool: **Interventions** combined with the **Principle of Independent Mechanisms**.
    
    We will demonstrate this on one of the ambiguous edges, $A - B$. The key insight is that our SCM uses 
    **Gaussian** noise, which creates perfect symmetry—this is why noise-based methods (like LiNGAM) also 
    fail on the observational data.
    """
)

st.subheader("Step 1: The Observational Ambiguity")
st.markdown(
    """
    We'll use the *same observational data* from the PC demo. We test the residuals for independence 
    in both directions (the LiNGAM/noise-based method from Page 3).
    
    **Remember:** Our data uses Gaussian noise: $N_A, N_B \\sim \\mathcal{N}(0, 1)$
    """
)

if 'obs_ambiguity' not in st.session_state:
    st.session_state.obs_ambiguity = None

if st.button("Run Observational Check on $A$ and $B$", use_container_width=True, key="obs_check_btn"):
    # Use the same data generated by the PC alg button
    if st.session_state.pc_results:
        df_obs_amb = st.session_state.pc_results["data"]
        
        # 2. Test A -> B
        res_A_B = sim_indep.fit_and_get_residuals(df_obs_amb, 'A', 'B')
        df_res_A_B = pd.DataFrame({'A': df_obs_amb['A'], 'Residuals': res_A_B})
        fig_A_B = charts.create_scatter_plot(df_res_A_B, 'A', 'Residuals', 'Test $A \\to B$: Residuals of B vs. A')
        
        # 3. Test B -> A
        res_B_A = sim_indep.fit_and_get_residuals(df_obs_amb, 'B', 'A')
        df_res_B_A = pd.DataFrame({'B': df_obs_amb['B'], 'Residuals': res_B_A})
        fig_B_A = charts.create_scatter_plot(df_res_B_A, 'B', 'Residuals', 'Test $B \\to A$: Residuals of A vs. B')
        
        st.session_state.obs_ambiguity = (fig_A_B, fig_B_A)
    else:
        st.error("Please run the PC Algorithm in Part 1 first to generate data.")

if st.session_state.obs_ambiguity:
    fig_A_B, fig_B_A = st.session_state.obs_ambiguity
    col1_obs, col2_obs = st.columns(2)
    with col1_obs:
        st.plotly_chart(fig_A_B, use_container_width=True)
    with col2_obs:
        st.plotly_chart(fig_B_A, use_container_width=True)
    
    st.warning(
        """
        **Result: We're Stuck!**
        
        Because the data uses **Gaussian noise** ($N_A, N_B \\sim \\mathcal{N}(0, 1)$), the residuals in 
        **both directions** are also Gaussian and appear independent of the cause.
        
        **Why both look the same:**
        - **Test $A \\to B$:** Fit $B = \\beta A + \\epsilon$. Residuals are $\\sim \\mathcal{N}(0, 1)$, independent 
        - **Test $B \\to A$:** Fit $A = \\beta B + \\epsilon$. Residuals are also $\\sim \\mathcal{N}(0, 1)$, independent 
        
        Gaussian distributions are **perfectly symmetric**, so linear regression creates Gaussian residuals 
        in both directions!
        
        Both the PC algorithm (due to Markov Equivalence) and the noise-based method (due to Gaussianity) 
        fail to find the direction from observational data alone.
        """
    )


st.subheader("Step 2: Breaking the Tie with an Intervention")
st.markdown(
    """
    Now, let's perform an experiment. We will **intervene on $A$**, changing its distribution from 
    Gaussian to **Uniform**: $do(A := \\text{Uniform}(3, 7))$.
    
    **This is a dramatic change:** We're replacing a bell-curved (Gaussian) distribution with a 
    flat rectangular (Uniform) distribution, while keeping the mechanism noise the same ($N_B \\sim \\mathcal{N}(0, 1)$).
    
    According to the **Principle of Independent Mechanisms**, the *true* causal mechanism ($B \\leftarrow A$) 
    should remain **invariant** (residuals still independent), while the *fake* anti-causal model 
    ($A \\leftarrow B$) should **break** (residuals show the rectangular shape).
    """
)

if 'int_ambiguity' not in st.session_state:
    st.session_state.int_ambiguity = None

if st.button("Perform Intervention: $do(A := \\text{Uniform}(3, 7))$", type="primary", use_container_width=True, key="int_check_btn"):
    # 1. Generate interventional data
    df_int = sim.generate_diamond_interventional_data(n_samples=2000)
    
    # 2. Test A -> B (Correct Model)
    res_A_B_int = sim_indep.fit_and_get_residuals(df_int, 'A', 'B')
    df_res_A_B_int = pd.DataFrame({'A_int': df_int['A'], 'Residuals_int': res_A_B_int})
    fig_A_B_int = charts.create_scatter_plot(df_res_A_B_int, 'A_int', 'Residuals_int', 'Test $A \\to B$: Residuals of B vs. A (Interventional)')
    
    # 3. Test B -> A (Incorrect Model)
    res_B_A_int = sim_indep.fit_and_get_residuals(df_int, 'B', 'A')
    df_res_B_A_int = pd.DataFrame({'B_int': df_int['B'], 'Residuals_int': res_B_A_int})
    fig_B_A_int = charts.create_scatter_plot(df_res_B_A_int, 'B_int', 'Residuals_int', 'Test $B \\to A$: Residuals of A vs. B (Interventional)')
    
    st.session_state.int_ambiguity = (fig_A_B_int, fig_B_A_int)

if st.session_state.int_ambiguity:
    fig_A_B_int, fig_B_A_int = st.session_state.int_ambiguity
    col1_int, col2_int = st.columns(2)
    with col1_int:
        st.plotly_chart(fig_A_B_int, use_container_width=True)
        st.success(
            """
            **Result 1: INVARIANT**
            
            The residuals of $B$ remain Gaussian, centered at 0, and independent of $A$—even though 
            $A$ is now drawn from a completely different distribution (Uniform instead of Gaussian)!
            
            **Why?** The mechanism $B := 1.0 \\cdot A + N_B$ (with $N_B \\sim \\mathcal{N}(0,1)$) is **invariant** 
            - The functional relationship $B := 1.0 \\cdot A + N_B$ hasn't changed
            - The noise term $N_B ~ N(0,1)$ is still the same Gaussian distribution
            - The independence $N_B ⊥⊥ A$ is still satisfied (noise independent of cause)
            
            The residuals show the **same pattern** as before because the noise term $N_B$ hasn't changed.
            
            This is strong evidence for $A \\to B$.
            """
        )
    with col2_int:
        st.plotly_chart(fig_B_A_int, use_container_width=True)
        st.error(
            """
            **Result 2: BROKEN **
            
            The residuals of $A$ now show a clear **rectangular/diamond pattern**—they are visibly 
            dependent on $B$!
            
            **Why?** The model $A := \\beta \\cdot B + \\epsilon$ is trying to explain $A \\sim \\text{Uniform}(3, 7)$ 
            (a flat rectangular distribution) using Gaussian residuals. But you can clearly see the rectangular 
            shape "bleeding through" in the residual plot—the model cannot absorb the Uniform distribution 
            into its Gaussian noise term.
            
            The "mechanism" $A \\leftarrow B$ was just a statistical artifact in the observational data 
            (where both were Gaussian). It **breaks** as soon as we change $A$ to a different distribution.
            
            This proves $A \\not\\leftarrow B$.
            """
        )

st.markdown("Finally, we would do a similar intervention for $A - C$. Afterwhich, we successfully recover the causal graph we started with!")

st.divider()
st.header("Grand Conclusion: A Strategy for Causal Discovery")
st.markdown(
    """
    You have now completed the full journey, from basic SCMs to automated causal discovery. This final page demonstrates a powerful and efficient workflow that combines all the concepts from this app.
    
    This workflow is a strategy for answering "What is the true causal graph?" by intelligently combining "cheap" observational data with "expensive" experimental data.
    """
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Step 1: Learn What You Can (Observational)")
        st.markdown(
            """
            First, we use "cheap" observational data, which is often abundant.
            
            -   **Run the PC Algorithm:** As shown in Part 1, this is our most robust tool. It makes the fewest assumptions (just Markov & Faithfulness). It does the "heavy lifting" by finding the graph's **skeleton** and all identifiable **v-structures**.
            -   **Result:** We get a **CPDAG** (e.g., our Diamond graph with undirected edges $A-B$ and $A-C$).
            
            This step is crucial: instead of $N^2$ possible relationships, the PC algorithm hands us a short, targeted list of *specific ambiguities* we need to solve.
            """
        )

with col2:
    with st.container(border=True):
        st.subheader("Step 2: Solve What You Must (Interventional)")
        st.markdown(
            """
            Now, we use "expensive" experimental data (interventions) to resolve the remaining ambiguities.
            
            -   **Target the Ambiguity:** Instead of randomly experimenting, the CPDAG from Step 1 tells us *exactly* where to look. We only need to design experiments to orient the remaining undirected edges (e.g., $A-B$).
            -   **Test for Invariance:** As shown in Part 2, we can intervene on $A$ and check if the mechanism for $B$ remains stable.
            
            This targeted intervention breaks the symmetry and identifies the true causal direction, $A \\to B$.
            """
        )

st.subheader("Why This Two-Step Workflow is Powerful")
st.markdown(
    """
    Interventions, like running a clinical trial, launching an A/B test, or intervening on a gene, are often costly, time-consuming, or even unethical. We want to do as few of them as possible.
    
    The workflow you've just seen is an efficient, hybrid approach:
    
    1.  **PC Algorithm (Observational):** Use this "cheap" method first to discover the "easy" parts of the graph and, most importantly, to **identify the precise, remaining ambiguities**.
    2.  **Targeted Interventions (Experimental):** Use this "expensive" method *only* where necessary to resolve the *specific* ambiguities identified by the PC algorithm.
    """
)
