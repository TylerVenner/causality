import streamlit as st
import pandas as pd
import src.simulations.d_separation_sim as sim
import src.plotting.charts as charts

st.set_page_config(
    page_title="Causal Markov Property",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ The Causal Markov Property & d-Separation")
st.markdown(
    """
    On the previous pages, we saw how SCMs allow us to *simulate* data and predict the outcomes of interventions and counterfactuals.
    
    But this raises a critical question: In the real world, we don't *know* the true causal graph. We just have data. How can we possibly discover the graph structure just by *looking* at observational data?
    
    This page introduces the theoretical foundation that makes this possible. The link between a causal graph and the data it produces is called the **Causal Markov Property**. This property is the "rulebook" that causal discovery algorithms (like the PC algorithm) use to reverse-engineer the graph from the data.
    """
)

st.subheader("The Graph is a Map of Independencies")
st.markdown(
    """
    Consider this not so simple causal graph for a health scenario. We will use this as our example.
    """
)

st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=circle, style="filled", fillcolor=lightblue];
        
        Z1 [label="Genetics"];
        Z2 [label="Lifestyle"];
        X [label="Cholesterol"];
        Y [label="Heart Disease"];
        Z3 [label="Medication"];
        
        Z1 -> X;
        Z1 -> Y;
        Z2 -> X;
        Z2 -> Y;
        X -> Y;
        Z3 -> Y;
    }
""")

st.info(
    """
    **Formal Definition: The Causal Markov Property**
    
    This property states that any variable in a graph is **statistically independent of its non-descendants, given its direct causes (its parents)**.
    """
)

with st.expander("Click here for a concrete breakdown of what this means for our 'Health' graph"):
    st.markdown(
        """
        Let's apply that formal rule to our graph:
        
        1.  **Consider the variable $X$ (Cholesterol):**
            * **Parents:** $Z_1$ (Genetics), $Z_2$ (Lifestyle)
            * **Non-Descendants:** $Z_3$ (Medication)
            * **CMP Claim:** The property says $X$ is independent of $Z_3$, given $Z_1$ and $Z_2$.
            * **Intuition:** This graph claims that once you know a person's genetics and lifestyle, finding out whether they take heart medication tells you *nothing new* about their cholesterol level. Why? Because the graph assumes Medication ($Z_3$) affects Heart Disease ($Y$) *directly*, not by first changing Cholesterol ($X$). This is a strong, testable statistical claim!
        
        2.  **Consider the variable $Z_3$ (Medication):**
            * **Parents:** (None)
            * **Non-Descendants:** $Z_1$ (Genetics), $Z_2$ (Lifestyle), $X$ (Cholesterol)
            * **CMP Claim:** The property says $Z_3$ is independent of $Z_1$, $Z_2$, and $X$, given its empty set of parents.
            * **Intuition:** This graph claims that Medication Perp \{Genetics, Lifestyle, Cholesterol\}. This means the decision to take medication is statistically independent of a person's genetics, lifestyle, and cholesterol levels. In the real world, this is likely *false* (high cholesterol *causes* people to take medication), which would mean our assumed graph is *wrong*.
        
        This is the key insight: the graph makes powerful, falsifiable claims about the data.
        """
    )

st.subheader('d-Separation: The "Read" Rules for Any Graph')
st.markdown(
    """
    The Causal Markov Property gives rise to a complete set of graphical rules called **d-separation** (which stands for "directional separation").
    
    These rules are the "decoding key" for the graph. They allow us to look at *any* three sets of nodes (X, Y, and Z) and ask, "Does this graph claim that $X$ is independent of $Y$ given $Z$?"
    
    The PC algorithm is just a clever program that:
    1.  Tests all these independencies in the *data*.
    2.  Finds the one graph (or set of graphs) whose d-separation rules exactly match the list of independencies it found.
    """
)

st.divider()

st.header("The 3 Junctions of d-Separation")
st.markdown(
    """
    We can understand all the rules of d-separation by looking at the three possible 3-variable structures. For each structure, we'll test for correlation in two scenarios:
    1.  **Observational:** A simple scatter plot of $X$ and $Y$.
    2.  **Conditional:** A scatter plot of the *residuals* of $X \sim Z$ vs. the *residuals* of $Y \sim Z$. This plot reveals the correlation between $X$ and $Y$ *after* we've "controlled for" or "adjusted for" $Z$.
    """
)

st.sidebar.header("Simulation Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 500, key="dsep_samples")

@st.cache_data
def cached_generate_data(structure_type, n_samples):
    return sim.generate_data(structure_type, n_samples)

# Tabs for the 3 Structures 
tab1, tab2, tab3 = st.tabs(["**Structure 1: The Chain (Mediation)**", "**Structure 2: The Fork (Confounding)**", "**Structure 3: The Collider (v-structure)**"])

with tab1:
    st.subheader("Structure 1: The Chain (Mediation)")
    st.graphviz_chart("digraph { rankdir=LR; X -> Z -> Y }")
    
    # SCM Expander for Chain 
    with st.expander("Show the underlying Structural Causal Model (SCM)"):
        st.latex(r'''
            \begin{aligned}
                N_X, N_Z, N_Y &\sim \mathcal{N}(0, 1) \quad (\text{independent}) \\
                \\
                X &:= N_X \\
                Z &:= 1.5 \cdot X + N_Z \\
                Y &:= 2.0 \cdot Z + N_Y
            \end{aligned}
        ''')
    
    st.markdown(
        """
        - **Rule:** The path from $X$ to $Y$ is **BLOCKED** if we condition on the mediator $Z$.
        - **Interpretation:** $X$ and $Y$ are correlated, but they are *conditionally independent* given $Z$. The information from $X$ flows *through* $Z$ to get to $Y$.
        - **Connection:** This explains **Rule 2** from the "Confounding vs. Mediation" page. Adjusting for a mediator blocks the causal path.
        """
    )
    
    df_chain = cached_generate_data('chain', n_samples)
    condition_chain = st.checkbox("Condition on Z (the Mediator)", value=False, key='chain')
    
    if condition_chain:
        res_x = sim.get_residuals(df_chain, 'X', 'Z')
        res_y = sim.get_residuals(df_chain, 'Y', 'Z')
        plot_df = pd.DataFrame({'X (Residuals)': res_x, 'Y (Residuals)': res_y})
        fig = charts.create_scatter_plot(plot_df, 'X (Residuals)', 'Y (Residuals)', 'X vs. Y (Conditioned on Z)')
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Result:** The correlation vanishes! By conditioning on the mediator $Z$, we have blocked the flow of information from $X$ to $Y$.")
    else:
        fig = charts.create_scatter_plot(df_chain, 'X', 'Y', 'Observational Data: X vs. Y')
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Result:** $X$ and $Y$ are clearly correlated, as expected.")

with tab2:
    st.subheader("Structure 2: The Fork (Confounding)")
    st.graphviz_chart("digraph { rankdir=LR; Z -> X; Z -> Y }")

    # SCM Expander for Fork 
    with st.expander("Show the underlying Structural Causal Model (SCM)"):
        st.latex(r'''
            \begin{aligned}
                N_Z, N_X, N_Y &\sim \mathcal{N}(0, 1) \quad (\text{independent}) \\
                \\
                Z &:= N_Z \\
                X &:= 1.5 \cdot Z + N_X \\
                Y &:= 2.0 \cdot Z + N_Y
            \end{aligned}
        ''')

    st.markdown(
        """
        - **Rule:** The "back-door" path from $X$ to $Y$ is **BLOCKED** if we condition on the confounder $Z$.
        - **Interpretation:** $X$ and $Y$ are correlated (spuriously), but they are *conditionally independent* given their common cause $Z$.
        - **Connection:** This explains **Rule 1** from the "Confounding vs. Mediation" page. Adjusting for a confounder blocks the non-causal path.
        """
    )
    
    df_fork = cached_generate_data('fork', n_samples)
    condition_fork = st.checkbox("Condition on Z (the Confounder)", value=False, key='fork')
    
    if condition_fork:
        res_x = sim.get_residuals(df_fork, 'X', 'Z')
        res_y = sim.get_residuals(df_fork, 'Y', 'Z')
        plot_df = pd.DataFrame({'X (Residuals)': res_x, 'Y (Residuals)': res_y})
        fig = charts.create_scatter_plot(plot_df, 'X (Residuals)', 'Y (Residuals)', 'X vs. Y (Conditioned on Z)')
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Result:** The (spurious) correlation vanishes! By conditioning on the common cause $Z$, we have blocked the non-causal path and correctly identified that there is no direct link between $X$ and $Y$.")
    else:
        fig = charts.create_scatter_plot(df_fork, 'X', 'Y', 'Observational Data: X vs. Y')
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Result:** $X$ and $Y$ are correlated due to their common cause $Z$.")

with tab3:
    st.subheader("Structure 3: The Collider (v-structure)")
    st.graphviz_chart("digraph { rankdir=LR; X -> Z; Y -> Z }")

    # SCM Expander for Collider 
    with st.expander("Show the underlying Structural Causal Model (SCM)"):
        st.latex(r'''
            \begin{aligned}
                N_X, N_Y, N_Z &\sim \mathcal{N}(0, 1) \quad (\text{independent}) \\
                \\
                X &:= N_X \\
                Y &:= N_Y \\
                Z &:= 2.0 \cdot X + 1.5 \cdot Y + N_Z
            \end{aligned}
        ''')

    st.markdown(
        """
        - **Rule:** This is the most important rule. The path between $X$ and $Y$ is **OPEN** *if and only if* we condition on the collider $Z$ (or a descendant of $Z$).
        - **Interpretation:** $X$ and $Y$ are *unconditionally independent* (they have no path between them). But if we learn the value of their common effect $Z$, we "open" a path.
        - **Example:** "Sprinkler" ($X$) and "Rain" ($Y$) are independent. But if we observe the "Grass is Wet" ($Z$), they become correlated. If the grass is wet and you know it *didn't* rain, you can deduce the sprinkler must have been on. This is **Berkson's Paradox**.
        """
    )
    
    df_collider = cached_generate_data('collider', n_samples)
    condition_collider = st.checkbox("Condition on Z (the Collider)", value=False, key='collider')
    
    if condition_collider:
        res_x = sim.get_residuals(df_collider, 'X', 'Z')
        res_y = sim.get_residuals(df_collider, 'Y', 'Z')
        plot_df = pd.DataFrame({'X (Residuals)': res_x, 'Y (Residuals)': res_y})
        fig = charts.create_scatter_plot(plot_df, 'X (Residuals)', 'Y (Residuals)', 'X vs. Y (Conditioned on Z)')
        st.plotly_chart(fig, use_container_width=True)
        st.error("**Result:** A correlation *appears*! By conditioning on the common effect $Z$, we have created a new, non-causal dependency between $X$ and $Y$.")
    else:
        fig = charts.create_scatter_plot(df_collider, 'X', 'Y', 'Observational Data: X vs. Y')
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Result:** $X$ and $Y$ are independent and uncorrelated, as expected. They are two separate causes.")

st.divider()

st.header("Putting It All Together: How d-Separation Works")
st.markdown(
    """
    The three junctions (Chain, Fork, Collider) are the building blocks. To find the relationship between *any* two nodes $A$ and $B$ in a large graph, given a set of nodes $Z$ to condition on, we just follow this simple recipe:
    
    1.  Find **every path** that exists between $A$ and $B$ (just trace the lines, ignoring arrowheads).
    2.  For each path, check if it is **blocked** by our conditioning set $Z$.
    3.  A path is **blocked** if:
        * It contains a **Chain** ($... \\to M \\to ...$) or a **Fork** ($... \leftarrow M \\to ...$) where the middle node $M$ is **IN** the set $Z$.
        * **OR** it contains a **Collider** ($... \\to M \leftarrow ...$) where the middle node $M$ is **NOT** in the set $Z$ (and neither are any of $M$'s descendants).
    
    If **ALL** paths between $A$ and $B$ are blocked, then the graph claims $A \perp\kern-5pt\perp B \mid Z$.
    """
)

st.subheader("Example: Applying the Rules to the 'Health' Graph")
st.markdown("Let's use these rules on our original graph to make two predictions:")

st.graphviz_chart("""
    digraph {
        rankdir=LR;
        node [shape=circle, style="filled", fillcolor=lightblue];
        
        Z1 [label="Genetics"];
        Z2 [label="Lifestyle"];
        X [label="Cholesterol"];
        Y [label="Heart Disease"];
        Z3 [label="Medication"];
        
        Z1 -> X;
        Z1 -> Y;
        Z2 -> X;
        Z2 -> Y;
        X -> Y;
        Z3 -> Y;
    }
""")

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    with st.container(border=True):
        st.markdown("**Test 1: Are Genetics ($Z_1$) and Lifestyle ($Z_2$) independent?**")
        st.latex(r"Z_1 \perp\kern-5pt\perp Z_2 \mid \emptyset")
        st.markdown(
            """
            1.  **Find Paths:** There is only one path: $Z_1 \\to X \leftarrow Z_2$.
            2.  **Check Path:** This path contains a **Collider** at $X$.
            3.  **Check Rule:** The conditioning set is empty ($\emptyset$), so the collider $X$ is **NOT** in the set.
            
            **Conclusion:** The path is **BLOCKED**. Since all paths (i.e., the one path) are blocked, the graph claims $Z_1$ and $Z_2$ are independent. This makes intuitive sense.
            """
        )
with col_ex2:
    with st.container(border=True):
        st.markdown("**Test 2: Are they independent *given* Cholesterol ($X$)?**")
        st.latex(r"Z_1 \perp\kern-5pt\perp Z_2 \mid X")
        st.markdown(
            """
            1.  **Find Paths:** Same path: $Z_1 \\to X \leftarrow Z_2$.
            2.  **Check Path:** This path contains a **Collider** at $X$.
            3.  **Check Rule:** The conditioning set is $\{X\}$, so the collider $X$ **IS** in the set.
            
            **Conclusion:** The path is **OPEN**. This is Berkson's Paradox! The graph claims that if you *don't* know a person's cholesterol, their genetics and lifestyle are independent. But if you *do* know they have high cholesterol, their genetics and lifestyle become correlated.
            """
        )

# Faithfulness Section 
st.header("The Other Half of the Pact: Faithfulness")
st.markdown(
    """
    The Causal Markov Property (which gives us d-separation) is only half the story. It states:
    
    **Graph $\implies$ Independencies**
    
    To do discovery, we need to go the other way. We need to assume the **Faithfulness** condition:
    
    **Independencies $\implies$ Graph**
    
    The Faithfulness assumption just states that *all* the independencies in the data are a result of the graph's structure (d-separation) and not just an accident.
    
    - **Example of Unfaithfulness:** Imagine a Chain $X \\to Z \\to Y$ where the causal effect is $+1$. At the same time, there is a direct path $X \\to Y$ with an effect of $-1$. These two paths would perfectly cancel each other out, and the data would show $X \perp\kern-5pt\perp Y$. This is an "unfaithful" distribution because the data has an independence that the graph structure doesn't imply.
    
    The PC algorithm *assumes faithfulness*—it assumes that such perfect cancellations don't happen.
    """
)

st.header("The Causal Pact: Markov & Faithfulness")
st.markdown(
    """
    The Causal Markov Property and the Faithfulness condition form the fundamental "pact" that connects a graph to data. The PC algorithm relies on both sides of this pact.
    """
)

col_markov, col_faith = st.columns(2)

with col_markov:
    with st.container(border=True):
        st.subheader("1. Causal Markov Property")
        st.markdown("**Graph $\implies$ Independencies**")
        st.markdown(
            """
            This is the "read" rule we've been using. It states that if your graph $\mathcal{G}$ is the true causal model for a probability distribution $P$, then $P$ *must* contain all the independencies that are implied by d-separation in $\mathcal{G}$.
            """
        )
        st.latex(r'''
            \text{For any disjoint sets } A, B, Z: \\
            A \perp\kern-5pt\perp_{\mathcal{G}} B \mid Z \implies A \perp\kern-5pt\perp_{P} B \mid Z
        ''')
        st.markdown("**In English:** If a path is blocked in the graph, you are guaranteed to find a statistical independence in the data.")

with col_faith:
    with st.container(border=True):
        st.subheader("2. Faithfulness Assumption")
        st.markdown("**Independencies $\implies$ Graph**")
        st.markdown(
            """
            This is the "reverse" rule, and it's an *assumption*. It states that the *only* independencies present in the distribution $P$ are the ones implied by the Causal Markov Property.
            """
        )
        st.latex(r'''
            \text{For any disjoint sets } A, B, Z: \\
            A \perp\kern-5pt\perp_{P} B \mid Z \implies A \perp\kern-5pt\perp_{\mathcal{G}} B \mid Z
        ''')
        st.markdown('**In English:** If you find a statistical independence in the data, you can assume it is because a path is blocked in the graph. The data is not "lying" with coincidental independencies.')


st.success(
    """
    **Final Takeaway:**
    
    - **Causal Markov Property:** Gives us the rules (d-separation) to read independencies *from* a graph.
    - **Faithfulness Assumption:** Lets us trust that the independencies we *find in data* correspond to d-separation in the *true graph*.
    
    Together, these two principles give us a **bi-directional link** between graph structure and statistical data, which is what allows an algorithm like PC to work.
    """
)
