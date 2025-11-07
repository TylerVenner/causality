# pages/1_🔬_Asymmetry_of_Interventions.py

import streamlit as st

from src.simulations.intervention_sim import generate_observational_data, perform_intervention
from src.plotting.charts import create_overlaid_density_plot, create_scatter_plot, create_comparison_density_plot

st.set_page_config(
    page_title="Simulation 1: Asymmetry of Interventions",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 The Asymmetry of Interventions")
st.markdown(
    """
    On the previous page, we introduced **Structural Causal Models (SCMs)** as a formal language for describing causal processes. Now, we'll see why this framework is so powerful through exploiting properties of **interventions**.
    """
)

st.divider()

# --- Section 2: The Fundamental Asymmetry of Causation ---
st.header("The Fundamental Asymmetry of Causation")
st.markdown(
    """
    This ability to perform local surgeries reveals the core asymmetry of cause and effect: **intervening on a cause can change its effect, but intervening on an effect does *not* change its cause.**
    
    Let's demonstrate this with the linear SCM from the textbook (Example 3.2).
    """
)
st.markdown(
    r"""
    **Ground Truth SCM ($\mathfrak{C}$):** Consider the causal graph $C \to E$ defined by the following linear model, where the noises $N_C$ and $N_E$ are independent standard normal variables ($\mathcal{N}(0,1)$).
    """
)
st.latex(r'''
    \mathfrak{C}: \begin{cases}
        C &:= N_C \\
        E &:= 4 \cdot C + N_E
    \end{cases}
''')

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Case 1: Intervening on the Cause")
    st.markdown("What happens if we intervene and set $C$ to 2?")
    st.latex(r"\text{do}(C := 2)")
    st.markdown(
        r"""
        We replace the first equation. The new SCM defines the post-intervention behavior of $E$. Substituting the new value of $C$ into the equation for $E$ yields $E := 4 \cdot (2) + N_E = 8 + N_E$.
        """
    )
    st.success(r"**Conclusion:** The interventional distribution $P_E^{\mathfrak{C};\text{do}(C:=2)}$ is $\mathcal{N}(8,1)$, a significant change from the observational distribution $P_E^{\mathfrak{C}} = \mathcal{N}(0,17)$.")

with col_b:
    st.subheader("Case 2: Intervening on the Effect")
    st.markdown("What happens if we intervene and set $E$ to 2?")
    st.latex(r"\text{do}(E := 2)")
    st.markdown(
        r"""
        We replace the second equation. The new SCM is:
        $C := N_C$
        
        $E := 2$
        
        Crucially, the mechanism for the cause, $C := N_C$, is **untouched**.
        """
    )
    st.success(r"**Conclusion:** The interventional distribution $P_C^{\mathfrak{C};\text{do}(E:=2)}$ is $\mathcal{N}(0,1)$, which is identical to the original observational distribution $P_C^{\mathfrak{C}}$.")

st.markdown(
    r"""
    This asymmetry highlights the crucial difference between intervention and conditioning. In the observational world, if we *see* that $E=2$, we can infer something about $C$. The conditional distribution $P_{C|E=2}^{\mathfrak{C}}$ is **not** the same as the original distribution of $C$.
    
    However, if we *force* $E$ to be 2 via an intervention, we break the original system. The interventional distribution $P_C^{\mathfrak{C};\text{do}(E:=2)}$ **is** the same as the original distribution $P_C^{\mathfrak{C}}$.
    """
)

st.divider()

st.header("Interactive Simulation")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000)
slope = st.sidebar.slider("Slope (b)", -5.0, 5.0, 2.0)

NOISE_STD = 1.5
COLOR_X = '#1f77b4'  
COLOR_Y = '#ff7f0e' 

st.subheader("1. The Ground Truth Model")
st.markdown("First, we define our ground truth Structural Causal Model (SCM).")

with st.container(border=True):
    st.markdown("**Structural Assignments:**")
    st.latex(fr'''
        \begin{{aligned}}
            N_X &\sim \mathcal{{N}}(0, 1) \\
            N_Y &\sim \mathcal{{N}}(0, {NOISE_STD**2:.2f}) \\
            \\
            X &:= N_X \\
            Y &:= {slope:.2f} \cdot X + N_Y
        \end{{aligned}}
    ''')

st.subheader("2. Observational Data & Theoretical Distributions")
st.markdown(
    """
    From this SCM, we can derive the theoretical distributions for $X$ and $Y$ that we would expect to see in the observational data.
    """
)
with st.expander("Show me the calculations"):
    st.markdown("**Distribution of X:**")
    st.latex("X := N_X \implies X \sim \mathcal{N}(0, 1)")
    
    st.markdown("**Distribution of Y:**")
    var_y = slope**2 + NOISE_STD**2
    st.latex(fr'''
        \begin{{aligned}}
        E[Y] &= E[b \cdot X + N_Y] = b \cdot E[X] + E[N_Y] = 0 \\
        \operatorname{{Var}}(Y) &= \operatorname{{Var}}(b \cdot X + N_Y) \\
        &= b^2 \cdot \operatorname{{Var}}(X) + \operatorname{{Var}}(N_Y)
        \quad (\text{{since $X$ and $N_Y$ are independent}}) \\
        &= ({slope:.2f})^2 \cdot 1^2 + {NOISE_STD**2:.2f} \\
        &= {var_y:.2f}
        \end{{aligned}}
    ''')

    # separate LaTeX for the distribution of Y:
    st.latex(fr"Y := {slope:.2f} \cdot X + N_Y \implies Y \sim \mathcal{{N}}(0, {var_y:.2f})")


# Calculate the variance for Y dynamically
var_y = slope**2 + NOISE_STD**2
st.success(f"**Theoretical Observational Distributions:** $X \\sim \\mathcal{{N}}(0, 1)$ and $Y \\sim \\mathcal{{N}}(0, {var_y:.2f})$")

@st.cache_data
def cached_generate_observational_data(samples, b):
    return generate_observational_data(n_samples=samples, slope=b)

@st.cache_data
def cached_perform_intervention(var_name, value, samples, b):
    return perform_intervention(var_name=var_name, value=value, n_samples=samples, slope=b)

# Generate and Plot Observational Data 
df_obs = cached_generate_observational_data(n_samples, slope)
col1_obs, col2_obs = st.columns(2)
with col1_obs:
    fig_scatter = create_scatter_plot(df_obs, 'X', 'Y', 'Empirical Observational Data')
    st.plotly_chart(fig_scatter, use_container_width=True)
with col2_obs:
    fig_obs_density = create_overlaid_density_plot(
        df_obs['X'], df_obs['Y'], 'Cause (X)', 'Effect (Y)', 'Empirical Observational Distributions'
    )
    st.plotly_chart(fig_obs_density, use_container_width=True)


st.subheader("3. Perform an Intervention")
st.markdown(
    """
    Now, press a button to perform a "surgical" intervention. The plot compares the **original** empirical distribution (dashed line) with the **new** distribution (solid line) after intervening on the other variable.
    """
)

st.markdown("#### A) Intervene on the Cause (X)")
st.write("What happens to the distribution of the Effect (Y) if we force X to be a specific value?")
col_btnx1, col_btnx2, col_btnx3 = st.columns(3)
intervention_values_x = [-2, 0, 2]
for i, val in enumerate(intervention_values_x):
    with locals()[f"col_btnx{i+1}"]:
        if st.button(f"**do(X := {val})**", use_container_width=True, key=f"btn_x_{val}"):
            df_int_x = cached_perform_intervention('X', val, n_samples, slope)
            fig_int_x = create_comparison_density_plot(
                df_obs['Y'], df_int_x['Y_post_intervention'], 'Original Y', f'Y after do(X={val})', 'Distribution of Y Shifts', color=COLOR_Y
            )
            st.plotly_chart(fig_int_x, use_container_width=True)
            with st.expander("Show me the theoretical calculation for this intervention"):
                st.markdown("The modified SCM becomes: $X := " + str(val) + r"$, so $Y := " + f"{slope:.2f} \\cdot {val} + N_Y$.")
                st.latex(fr"Y \sim \mathcal{{N}}({slope * val:.2f}, {NOISE_STD**2:.2f})")
                st.info("Notice how the empirical result (solid line) centers on the new theoretical mean.")

st.markdown("#### B) Intervene on the Effect (Y)")
st.write("What happens to the distribution of the Cause (X) if we force Y to be a specific value?")
col_btny1, col_btny2, col_btny3 = st.columns(3)
intervention_values_y = [-5, 0, 5]
for i, val in enumerate(intervention_values_y):
    with locals()[f"col_btny{i+1}"]:
        if st.button(f"**do(Y := {val})**", use_container_width=True, key=f"btn_y_{val}"):
            df_int_y = cached_perform_intervention('Y', val, n_samples, slope)
            fig_int_y = create_comparison_density_plot(
                df_obs['X'], df_int_y['X_post_intervention'], 'Original X', f'X after do(Y={val})', 'Distribution of X is Unchanged', color=COLOR_X
            )
            st.plotly_chart(fig_int_y, use_container_width=True)
            with st.expander("Show me the theoretical calculation for this intervention"):
                st.markdown("The modified SCM becomes: $Y := " + str(val) + r"$, but the mechanism for $X$ is untouched.")
                st.latex(r"X := N_X \implies X \sim \mathcal{N}(0, 1)")
                st.info("The theory predicts no change in the distribution of X, which is exactly what the simulation shows.")


st.header("The Significance of Asymmetry: Why This Matters")
st.markdown(
    """
    This asymmetry is the key to distinguishing cause from effect and is the conceptual basis for many causal discovery algorithms.
    """
)

st.subheader("Revisiting First Principles: Changing the Distribution Itself")
st.markdown(
    r"""
    Let's tie this back to the core idea from our introduction.
    
    -   **The Machine Learning Paradigm** assumes we are always working with a **single, fixed joint distribution**, let's call it $P_{obs}(X,Y)$. All statistical operations (like calculating $E[Y|X=x]$) are queries about this one distribution. The world is static.
    
    -   **The Causal Paradigm** recognizes that an intervention is an action that **changes the system itself**. When we perform an action like $do(X:=c)$, we are breaking the old rules. This act of intervention *destroys* the original distribution $P_{obs}(X,Y)$ and creates an entirely **new one**, $P_{do(X:=c)}(X,Y)$.
    
    The asymmetry we observed is a property of this change.
    
    -   When we intervened on the cause ($do(X:=c)$), we created a new world where the distribution of $Y$ was different.
    -   When we intervened on the effect ($do(Y:=c)$), we also created a new world, but in this specific new world, the distribution of $X$ happened to be the same as it was in the old one.
    
    This is the crucial insight: a causal model is not a model of one distribution, but a model of how the distribution **changes** in response to interventions.
    """
)

st.success(
    """
    **Takeaway:** The asymmetry of interventions is the empirical key that unlocks the difference between mere correlation and true causation. It allows us to test our causal hypotheses and understand not just what *is*, but what *would be* if we were to act.
    """
)