import streamlit as st
import pandas as pd
import networkx as nx
import pingouin as pg
import graphviz
from itertools import combinations
from typing import Dict, Set, Tuple, List

import src.simulations.fci_simulation as sim_fci
import src.plotting.charts as charts
from src.algorithms.pc_algorithm import pc_step_2_orient_colliders, pc_step_3_orient_remaining

# LOCAL PC ALGORITHM WITH LOGGING (for this page only) 

def _get_neighbors(graph: nx.Graph, node) -> Set:
    """Helper to get the set of current neighbors for a node."""
    return set(graph.neighbors(node))

def _partial_correlation_test(data: pd.DataFrame, i: str, j: str, S: Set[str], alpha: float, log: List[str]) -> bool:
    """Performs CI test and logs results."""
    n_samples = len(data)
    if n_samples < len(S) + 3:
        log.append(f"SKIPPED: {i} _||_ {j} | {S} (n_samples too small)")
        return False

    try:
        result = pg.partial_corr(data=data, x=i, y=j, covar=list(S) if S else None)
        p_value = result['p-val'].iloc[0]
        is_independent = p_value > alpha
        verdict = "INDEPENDENT" if is_independent else "Dependent"
        log.append(f"Test: {i} _||_ {j} | {S}?  p-val: {p_value:.4f} > {alpha}.  Verdict: {verdict}")
        return is_independent
    except Exception as e:
        log.append(f"ERROR: {i} _||_ {j} | {S}. Error: {e}")
        return False

def _pc_step_1_with_logging(data: pd.DataFrame, alpha: float) -> Tuple[nx.Graph, Dict, List[str]]:
    """Local PC skeleton function with logging."""
    nodes = list(data.columns)
    skeleton = nx.complete_graph(nodes)
    sepset = {}
    log = []
    
    k = 0
    while True:
        k_changed = False
        log.append(f"--- Testing with conditioning set size k = {k} ---")
        edges_to_remove = []
        
        for (i, j) in list(skeleton.edges()):
            adj_i = _get_neighbors(skeleton, i) - {j}
            adj_j = _get_neighbors(skeleton, j) - {i}
            adj_set = adj_i if len(adj_i) <= len(adj_j) else adj_j
            
            if len(adj_set) >= k:
                for S in combinations(adj_set, k):
                    if _partial_correlation_test(data, i, j, set(S), alpha, log):
                        edges_to_remove.append((i, j))
                        sepset[(i, j)] = set(S)
                        sepset[(j, i)] = set(S)
                        k_changed = True
                        log.append(f"REMOVING edge {i} -- {j} based on S = {S}")
                        break
        
        skeleton.remove_edges_from(edges_to_remove)
        k += 1
        
        if all(len(_get_neighbors(skeleton, n)) < k for n in nodes):
            log.append(f"Stopping: No node has {k} neighbors left.")
            break
            
    log.append("--- Skeleton search complete ---")
    return skeleton, sepset, log

st.set_page_config(
    page_title="Hidden Confounding & FCI",
    page_icon="👻",
    layout="wide"
)

st.title("👻 The Specter of Hidden Confounding")
st.markdown(
    """
    Up to this point, our entire app has made a giant, unspoken assumption: **Causal Sufficiency**.
    
    **Causal Sufficiency** is the assumption that we have measured *all common causes* (confounders) of the variables in our dataset.
    
    -   On Page 4 ("Confounding vs. Mediation"), we *could* find the true effect of Ads on Sales because we *measured* the Holiday Season confounder.
    -   On Page 6 ("PC Algorithm"), the algorithm worked because our SCM was causally sufficient. When it tested $B \perp\kern-5pt\perp C$, it was able to find and condition on their common cause $A$.
    
    This page asks the critical question: **What happens when the confounder is hidden?**
    """
)
st.warning(
    """
    **Spoiler:** The PC Algorithm will fail. It will either give a wrong answer or, at best, an incomplete one. 
    We will then introduce the **FCI (Fast Causal Inference)** algorithm, a more advanced method designed to handle this exact problem.
    """
)

st.info(
    """
    Full transparancy: I did not implmenet the FCI algorithm, so that you see is what I expect the algorithm to output. However, I designed it for pedagogical reasons.
    """
)
st.divider()

st.header("Interactive Demo: How the PC Algorithm Fails")
st.markdown(
    """
    We will test the PC algorithm on a "Bow-Tie" graph with **two hidden variables**. 
    $H_1$ is a hidden parent of $A$. $H_2$ is a hidden *confounder* of $D$ and $E$.
    The algorithm will only see the observed data for $A, B, C, D, E$.
    """
)

# Use session state to store results
if 'fci_results' not in st.session_state:
    st.session_state.fci_results = None

if st.button("Run PC Algorithm on Observed Data (A, B, C, D, E)", type="primary", use_container_width=True):
    with st.spinner("Generating confounded data and running PC algorithm..."):
        # 1. Generate data (A, B, C, D, E are returned)
        data = sim_fci.generate_m_graph_data(n_samples=2000)
        
        # 2. Run PC Algorithm on the observed data WITH LOGGING
        skeleton, sepset, log = _pc_step_1_with_logging(data, alpha=0.05)
        pdag1 = pc_step_2_orient_colliders(skeleton.copy(), sepset)
        pdag_final = pc_step_3_orient_remaining(pdag1.copy())
        
        # 3. Get the "oracle" graphs
        true_graph_dot = sim_fci.get_m_graph_ground_truth_dot()
        fci_graph_dot = sim_fci.get_fci_correct_output_dot()
        pc_output_chart = charts.graphviz_from_nx(pdag_final, "PC Algorithm Output")

        # Store results
        st.session_state.fci_results = {
            "true_graph": true_graph_dot,
            "pc_output": pc_output_chart,
            "fci_output": fci_graph_dot,
            "log": log
        }

if st.session_state.fci_results:
    res = st.session_state.fci_results
    
    st.subheader("Results")
    tab_truth, tab_pc, tab_fci, tab_log = st.tabs([
        "1. Ground Truth (with Hidden Nodes)", 
        "2. PC Algorithm's (Incorrect) Output", 
        "3. FCI's (Correct) Output",
        "4. PC Algorithm Log"
    ])
    
    with tab_truth:
        st.markdown("This is the *true* SCM that generated the data. The nodes $H_1$ and $H_2$ are **hidden**; the algorithm only sees $A, B, C, D, E$.")
        st.graphviz_chart(res["true_graph"])
        with st.expander("Show Ground Truth SCM"):
            st.latex(r'''
                \begin{aligned}
                    H_1 &\sim \mathcal{N}(0, 1) \quad \text{(Hidden)} \\
                    H_2 &\sim \mathcal{N}(0, 1) \quad \text{(Hidden)} \\
                    A &:= 2.0 \cdot H_1 + N_A \\
                    B &:= 1.5 \cdot A + N_B \\
                    C &:= 1.0 \cdot B + N_C \\
                    D &:= 1.5 \cdot H_2 + N_D \\
                    E &:= 1.5 \cdot H_2 + 2.0 \cdot C + N_E
                \end{aligned}
            ''')
        
        with st.expander("🔍 Understanding the Causal Structure"):
            st.markdown(
                r"""
                This graph has several key features that will challenge the PC algorithm: 

                1.  **Causal Chain:** $A \to B \to C$. This is a simple, unconfounded chain. 
                2.  **Hidden Parent:** $H_1 \to A$. $H_1$ is unobserved, but it's *not* a confounder. It just adds noise to $A$. This part is not a problem.
                3.  **Hidden Confounder:** $D \leftarrow H_2 \to E$. $H_2$ is a classic unobserved common cause. This creates a spurious correlation between $D$ and $E$.$
                4.  **Collider:** $C \to E \leftarrow H_2$. The node $E$ is a collider.
                
                **The Critical Interaction:**
                
                The most important part is the interaction between features 3 and 4. There is a path from $C$ to $D$:
                
                $C \to E \leftarrow H_2 \to D$
                
                This path is **blocked** by the collider $E$. This means $C$ and $D$ are **marginally independent** ($C \perp\kern-5pt\perp D$).
                
                However, if we *condition* on the collider $E$, this path **opens up**. This means $C$ and $D$ are **conditionally dependent** ($C \not\perp\kern-5pt\perp D \mid E$).
                
                This is the classic signature of a v-structure! As we'll see, the PC algorithm will spot this signature... and draw the wrong conclusion.
                """
            )
            
    with tab_pc:
        st.markdown("This is the graph the **PC Algorithm** discovers. It assumes Causal Sufficiency, so it cannot represent the hidden confounders $H_1$ or $H_2$.")
        st.graphviz_chart(res["pc_output"])
        
        with st.expander("📊 Analyzing PC's Mistake (A Step-by-Step Log Analysis)"):
            st.markdown(
                r"""
                The PC algorithm fails in a specific, predictable way. Your logs show exactly how.
                
                **Step 1: Skeleton Discovery (Confirmed by Logs)**
                
                The algorithm runs conditional independence (CI) tests to find the skeleton.
                
                **At $k=0$ (Marginal Independence):**
                
                * `Test: C _||_ D | set()?  p-val: 0.8910 > 0.05.  Verdict: INDEPENDENT`
                * **Logic:** The algorithm correctly finds $C \perp D$.
                * **Why?** The true causal path is $C \to E \leftarrow H_2 \to D$. This path is **blocked** by the collider $E$. Since this is the only path connecting them, $C$ and $D$ are (correctly) found to be marginally independent. The edge $C-D$ is removed.
                * The logs also show $A \perp D$ and $B \perp D$, removing those edges.
                
                **At $k=1$ (Conditional Independence):**
                
                * `Test: A _||_ C | {'B'}?  p-val: 0.9864 > 0.05.  Verdict: INDEPENDENT`
                * **Logic:** The algorithm finds $A \perp C \mid B$.
                * **Why?** The true path is $A \to B \to C$. Conditioning on the mediator $B$ blocks this path. The edge $A-C$ is removed.
                
                * `Test: B _||_ E | {'C'}?  p-val: 0.3735 > 0.05.  Verdict: INDEPENDENT`
                * **Logic:** The algorithm finds $B \perp E \mid C$.
                * **Why?** The true path is $B \to C \to E$. Conditioning on the mediator $C$ blocks this path. The edge $B-E$ is removed.

                * `Test: A _||_ E | {'C'}?  p-val: 0.6518 > 0.05.  Verdict: INDEPENDENT`
                * **Logic:** The algorithm finds $A \perp E \mid C$.
                * **Why?** The true path is $A \to B \to C \to E$. Conditioning on $C$ blocks this path. The edge $A-E$ is removed.
                
                **Final Skeleton:**
                After all tests, the only edges *not* removed are $A-B$, $B-C$, $C-E$, and $D-E$.
                The algorithm correctly finds the skeleton: **$A - B - C - E - D$**.
                
                ---
                
                **Step 2: The Critical Error (Collider Orientation)**
                
                Now, PC applies its rule for orienting v-structures:
                
                > For any non-adjacent nodes $i$ and $j$ with a common neighbor $k$ (e.g., $i - k - j$):
                > Orient $i \to k \leftarrow j$ **if and only if** $k$ is **NOT** in the separating set $sepset(i, j)$.
                
                Let's trace this rule with our skeleton and logs:
                
                * **Test Case 1: $A - B - C$**
                    * Triplet: $i=A, k=B, j=C$.
                    * Log: `REMOVING edge A - C based on S = ('B',)`.
                    * This means $sepset(A, C) = \{'B'\}$.
                    * The common neighbor $B$ **IS** in the sepset. The rule does **not** apply. This triplet is left unoriented.
                
                * **Test Case 2: $B - C - E$**
                    * Triplet: $i=B, k=C, j=E$.
                    * Log: `REMOVING edge B - E based on S = ('C',)`.
                    * This means $sepset(B, E) = \{'C'\}$.
                    * The common neighbor $C$ **IS** in the sepset. The rule does **not** apply.
                
                * **Test Case 3: $C - E - D$ (The Critical Error)**
                    * Triplet: $i=C, k=E, j=D$.
                    * Log: `REMOVING edge C - D based on S = ()`.
                    * This means $sepset(C, D) = \{\}$ (an empty set).
                    * The common neighbor $E$ is **NOT** in the empty sepset.
                    * **The rule applies!** The algorithm confidently orients the v-structure: **$C \to E \leftarrow D$**.
                
                **This is the mistake.**
                
                The algorithm correctly identified the *statistical pattern* of a v-structure ($C \perp D$ and $C \not\perp D \mid E$), but it attributes it to the wrong *causal structure*.
                
                * **PC's Conclusion:** $C$ causes $E$, and $D$ *also* causes $E$.
                * **The Truth:** $C$ causes $E$, but $D$ is *confounded* with $E$ by $H_2$.
                
                The true structure $C \to E \leftarrow H_2 \to D$ produces the *exact same* statistical pattern. Because of its **Causal Sufficiency assumption**, PC cannot represent $H_2$ and defaults to the simplest (and in this case, incorrect) explanation.
                
                ---
                
                **Step 3: Propagation Fails (As It Should!)**
                
                However, our Meek rules are **conservative**: they only orient an edge if the alternative would create a *guaranteed* contradiction (a new v-structure or a cycle).
                
                Let's trace the logic for the unoriented triplet $A - B - C$:
                
                1.  After Step 2, the algorithm has the graph: $A - B - C \to E \leftarrow D$.
                2.  It considers the triplet $B - C - E$. We know $B$ and $E$ are non-adjacent.
                3.  The algorithm now implicitly "tests" the two possible orientations for the $B-C$ edge:
                    * **Possibility A:** Orient $B \to C$. This creates $B \to C \to E$. This is a valid chain and creates no new v-structures or cycles.
                    * **Possibility B:** Orient $B \leftarrow C$. This creates $B \leftarrow C \to E$. This is a valid *fork* and also creates no new v-structures or cycles.
                
                **This is the key:**
                
                Since both $B \to C$ and $B \leftarrow C$ are **equally consistent** with the conditional independence facts gathered so far, the PC algorithm has **no logical basis to choose one over the other.**
                
                Because it cannot make a choice, the conservative and correct action is to **leave the $B - C$ edge unoriented.**
                
                * Since $B - C$ remains unoriented, there is no orientation to propagate "down the line" to the $A - B$ edge. That also remains unoriented.
                
                The final, correct graph that the PC algorithm outputs is **$A - B - C \to E \leftarrow D$**. The algorithm has (correctly) been fooled by $H_2$ into creating the $D \to E$ link, and (correctly) determined it doesn't have enough information to orient the $A-B-C$ chain.
                
                > **A Key Lesson: PC vs. FCI**
                > 
                > What you've discovered is a core difference between PC and FCI, which perfectly reinforces the lesson on this page:
                > 
                > * **The PC Algorithm** (what you ran) correctly finds the v-structure $C \to E \leftarrow D$ but **stops** because it cannot resolve the ambiguity of the $A-B-C$ chain.
                > * **The FCI Algorithm** (as shown in your "correct" graph) has a more powerful and complex set of orientation rules. It can use other information (like the fact that $E$ is part of a confounded system) to resolve the ambiguity and correctly orient $A \to B \to C$.
                """
            )
        
        st.error(
            """
            **Analysis: FAILURE!**
            
            The PC algorithm has been fooled by the hidden confounder $H_2$. Because it *must* assume Causal Sufficiency,
            it incorrectly concludes that **$D$ is a cause of $E$**.
            
            This output is dangerously misleading. It gives a precise, simple, and **wrong** causal answer. It has missed the hidden confounder and invented a causal link that doesn't exist.
            """
        )

    with tab_fci:
        st.markdown("This is the **Partial Ancestral Graph (PAG)** that the **FCI Algorithm** would discover from the *exact same data*.")
        st.graphviz_chart(res["fci_output"])
        
        with st.expander("🎯 Understanding FCI's Output (A Partial Ancestral Graph)"):
            st.markdown(
                r"""
                The FCI algorithm does *not* assume Causal Sufficiency. It uses a richer graphical language—producing a **Partial Ancestral Graph (PAG)**—to represent its uncertainty.
                
                Let's break down the features of this (correct) graph:
                
                **1. $D \leftrightarrow E$ (Bi-directed Edge)**
                
                * **This is the 'smoking gun' for hidden confounding.**
                * **Why it happens:** The PC algorithm saw the v-structure pattern ($C \perp D$ and $C \not\perp D \mid E$) and incorrectly concluded $D \to E$.
                * FCI's more advanced rules test the *robustness* of this v-structure. It finds that the statistical pattern is **ambiguous**—it could be $D \to E$, *or* it could be $D \leftarrow H_2 \to E$.
                * Since FCI cannot be certain that $D$ causes $E$ (and, in fact, suspects it doesn't), it **rejects** PC's simple $D \to E$ arrow.
                * It replaces it with $D \leftrightarrow E$, which explicitly means: **"There exists an unobserved common cause (a latent confounder) of $D$ and $E$."** This correctly identifies the role of $H_2$.
                
                **2. $A \to B \to C \to E$ (Directed Edges)**
                
                * **This is the critical difference from PC.** You correctly noted that the PC algorithm **failed** to orient the $A -- B -- C$ chain.
                * **Unlike PC**, FCI has a more complete and powerful set of orientation rules. Once it has correctly identified the $D \leftrightarrow E$ confounded structure, it can propagate orientations throughout the *un*confounded parts of the graph with more confidence.
                * Its rules are powerful enough to resolve the ambiguity that PC got stuck on, allowing it to correctly orient the entire chain $A \to B \to C \to E$.

                **3. What About $H_1$? (A Note on Identifiability)**
                
                * You may have noticed the algorithm did *nothing* about $H_1$. It is completely invisible in the final graph.
                * This is **not a failure** of FCI. It is a fundamental limitation of identifiability.
                * **Why?** Algorithms like FCI detect hidden variables by the *statistical signatures* they create between **two or more** observed variables.
                * $H_2$ was detectable because it was a **confounder** ($D \leftarrow H_2 \to E$), creating a spurious correlation between $D$ and $E$.
                * $H_1$ is only a **latent source** ($H_1 \to A$). It doesn't create a relationship between $A$ and any *other* observed variable. Its influence is statistically indistinguishable from $A$'s own random noise term ($N_A$).
                * FCI's job is to find the ancestral relationships between the variables *we can see* and to flag *confounding* that would mislead us. It has done both perfectly.
                """
            )
        
        st.success(
            r"""
            **Analysis: SUCCESS!**
            
            FCI produces a graph that is "less precise" than PC's, but it is **correct and honest.**
            
            * It **correctly identifies** the hidden confounder $H_2$ by using the $D \leftrightarrow E$ edge.
            * It **correctly orients** the unconfounded part of the graph $A \to B \to C \to E$.
            * It **correctly ignores** the unidentifiable $H_1$, as it does not impact the causal relationships between the observed variables.
            
            FCI trades the **false precision** of the PC algorithm for **robust, honest uncertainty**.
            """
        )

    with tab_log:
        st.subheader("PC Algorithm Conditional Independence Test Log")
        st.markdown("This log shows every conditional independence test the PC algorithm performed. Look for the critical tests on $C, D, E$!")
        st.code("\n".join(res["log"]), language="text")
        
else:
    st.info("Click the button to generate confounded data and run the PC algorithm.")

st.divider()
st.header("👻 FCI: The Algorithm for Hidden Confounders")

st.markdown(
    r"""
    Our demo showed that the PC algorithm, when faced with a hidden confounder,
    will confidently give a **wrong answer**. It is fundamentally limited
    by its **Causal Sufficiency** assumption.
    
    This is the exact problem the **FCI (Fast Causal Inference)** algorithm
    was designed to solve.
    """
)

st.subheader("How FCI Works (Conceptually)")
st.markdown(
    r"""
    FCI is a more complex and robust algorithm that operates in a similar
    way to PC (finding a skeleton, then orienting edges), but with two
    critical differences:
    
    **1. It Does NOT Assume Causal Sufficiency**
    FCI is designed to work in the "real world" where you *must* assume
    that hidden confounders might exist.
    
    **2. It Uses a Richer Graphical Language: The PAG**
    The PC algorithm outputs a **DAG** (or a PDAG), which can only
    show simple directed edges ($A \to B$). FCI outputs a
    **PAG (Partial Ancestral Graph)**, which uses a larger set of symbols
    to represent uncertainty:
    
    * **$A \to B$ (Arrow):** $A$ is an ancestor of $B$ (and $B$ is not an
        ancestor of $A$).
    * **$A \leftrightarrow B$ (Bi-directed):** The "smoking gun" we saw.
        This explicitly means "There is a hidden confounder of $A$ and $B$."
    * **$A \circ\to B$ (Circle-Arrow):** A common FCI output. The circle $\circ$
        at the tail means "I am uncertain about this endpoint." It could be
        $A \to B$ or $A \leftrightarrow B$.
    * **$A \circ- \circ B$ (Circle-Circle):** The algorithm is uncertain
        about both endpoints.
    
    **3. It Has a More Powerful Rule Set**
    To orient these complex edges, FCI uses a much larger and more robust
    set of orientation rules (often 10 or more, compared to PC's 3-4).
    These rules are designed to test the *robustness* of a v-structure
    before orienting it, which is how it correctly catches the ambiguity
    that PC misses.
    """
)

st.subheader("PC vs. FCI: A Comparison")

st.markdown(
    r"""
    This reveals a fundamental tradeoff in causal discovery.
    
    | Feature | 🖥️ PC Algorithm (Fast, Precise) | 👻 FCI Algorithm (Slow, Robust) |
    | :--- | :--- | :--- |
    | **Key Assumption** | **Causal Sufficiency** (All confounders are measured) | **Causal Insufficiency** (Allows for hidden confounders) |
    | **Output Graph** | **PDAG** (Partial DAG) | **PAG** (Partial Ancestral Graph) |
    | **Edge Types** | $\to$, $-$ | $\to$, $\leftrightarrow$, $\circ\to$, $\circ-\circ$, etc. |
    | **Strength** | Very fast. Gives simple, precise answers. | **Honest.** Correctly identifies what *can't* be known. |
    | **Weakness** | Can be **confidently wrong** if a confounder is hidden. | Slower. Output is complex and "less certain" (by design). |
    | **When to Use** | Controlled experiments. Systems you *know* are simple. | Observational data (e.g., economics, biology, social science). |

    """
)

st.info(
    r"""
    **The Bigger Picture:** Your journey has covered:
    1.  **Basic concepts** (interventions, confounding)
    2.  **Causal discovery** (PC algorithm under ideal conditions)
    3.  **Real-world limitations** (hidden confounding and FCI)
    
    The field of causal inference is about navigating the gap between what
    we *want* to know (the true DAG) and what we *can* know from limited,
    imperfect data. PC and FCI represent two different points on this spectrum
    of assumptions vs. guarantees.
    """
)