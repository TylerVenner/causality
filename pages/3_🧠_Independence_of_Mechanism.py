import streamlit as st
import pandas as pd
from src.simulations.independence_sim import generate_data, fit_and_get_equation, generate_lingam_data, fit_and_get_residuals
from src.plotting.charts import create_scatter_plot

st.set_page_config(
    page_title="Principle of Independent Mechanisms",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 The Principle of Independent Mechanisms (PIM)")
st.markdown(
    """
    So far, we have seen that we need a causal model to predict the effect of **interventions**. But what if we don't know the correct causal direction? Can we discover it from **observational data alone**? This page introduces the core assumption that often allows us to do just that.
    """
)

st.header("The Ambiguity of Statistical Models")
st.markdown(
    r"""
    Let's continue with an example of **Fertilizer ($F$)** and **Crop Yield ($Y$)**. From observation, we see a strong positive correlation.
    
    By the definition of conditional probability, the joint distribution $p(y, f)$ can always be factored in two equivalent ways. Each factorization, however, tells a different *causal story*.
    """
)
col1_math, col2_math = st.columns(2)
with col1_math:
    st.markdown("<center><b>Story 1: Fertilizer causes Yield</b></center>", unsafe_allow_html=True)
    st.latex(r"(F \to Y)")
    st.latex("p(y,f) = p(y|f)p(f)")
    st.markdown(
        """
        This aligns with the story that a farmer chooses an amount of fertilizer to use (drawn from the distribution of choices $p(f)$), and then a separate, physical mechanism of nature determines the yield based on that choice (the conditional distribution $p(y|f)$).
        """
    )

with col2_math:
    st.markdown("<center><b>Story 2: Yield causes Fertilizer</b></center>", unsafe_allow_html=True)
    st.latex(r"(Y \to F)")
    st.latex("p(y,f) = p(f|y)p(y)")
    st.markdown(
        """
        This aligns with a strange story where a certain yield is chosen by nature (from $p(y)$), and then that yield somehow determines how much fertilizer the farmer *had* to have used (the mechanism $p(f|y)$).
        """
    )

st.warning("Both factorizations are mathematically identical and can perfectly describe any dataset. From a purely statistical view, there is no way to prefer one over the other. However, each view tells a different causal story.")

st.divider()

st.header("A Solution: The Principle of Independent Mechanisms (PIM)")
st.markdown(
    """
    Our intuition that the first story is more plausible is based on a deep principle. If the true causal structure is $F \\to Y$, we are assuming the data-generating process is composed of two distinct and independent modules.
    """
)

st.info(
    """
    -   **The Cause Distribution ($p(f)$):** This describes the distribution of the cause variable. In our case, this is the set of economic and psychological factors that govern a farmer's decision-making. This process takes place inside the farmer's head and is influenced by market prices, capital, and their beliefs about crop biology.
    -   **The Causal Mechanism ($p(y|f)$):** This describes the physical or biological process that generates the effect from the cause. Here, it represents the **crop's biological response** to a given amount of fertilizer.
    """
)

st.markdown(
    """
    The **Principle of Independent Mechanisms** formalizes this separation by postulating that for the correct causal direction:
    
    1.  **Autonomy and Invariance:** The distribution of the cause ($p(f)$) and the mechanism producing the effect from the cause ($p(y|f)$) are **autonomous**. This means the crop's biological response contains no information about the farmer's economic decisions, and vice-versa. The mechanism is an independent, modular piece of the world that is **invariant** to changes in the distribution of its inputs.
    
    2.  **Possibility of Independent Intervention:** It is in principle possible to perform a "surgical" intervention on the cause variable (e.g., convincing the farmer to try a new fertilizer strategy, thus changing $p(f)$) **without** changing the conditional distribution $p(y|f)$ that represents the crop's biology.
    
    In the anti-causal direction ($Y \\to F$), we do not believe the conditional $p(f|y)$ represents a physical mechanism, so we do not expect it to be invariant if the distribution of yields were to change.
    """
)

st.markdown(
    """
    **Our Setup:** We consider two different datasets, one from **"Small Farms" ($sf$)** and one from **"Industrial Farms" ($if$)**.
    - The overall data distributions are different: $p^{sf}(y,f) \\neq p^{if}(y,f)$.
    - This is because the distribution of fertilizer choices is different: $p^{sf}(f) \\neq p^{if}(f)$.
    """
)

col_test1, col_test2 = st.columns(2)

with col_test1:
    st.subheader("Testing the Causal Direction ($F \\to Y$)")
    st.markdown(
        r"""
        If $F \to Y$ is the correct causal structure, the PIM suggests that the physical mechanism connecting fertilizer to yield, $p(y|f)$, should be the same in both environments, as it is governed by crop biology. The distribution of causes, $p(f)$, can change.
        
        The two joint distributions should therefore factor as:
        """
    )
    st.latex(r'''
        \begin{aligned}
            p^{sf}(y,f) &= \mathbf{p(y|f)} \cdot p^{sf}(f) \\
            p^{if}(y,f) &= \mathbf{p(y|f)} \cdot p^{if}(f)
        \end{aligned}
    ''')
    st.success("Here, we have an **invariant conditional** term and a **changing marginal** term.")

with col_test2:
    st.subheader("Testing the Anti-Causal Direction ($Y \\to F$)")
    st.markdown(
        r"""
        If we were to factor the distributions in the anti-causal direction, we would have:
        """
    )
    st.latex(r'''
        \begin{aligned}
            p^{sf}(y,f) &= p^{sf}(f|y) \cdot p^{sf}(y) \\
            p^{if}(y,f) &= p^{if}(f|y) \cdot p^{if}(y)
        \end{aligned}
    ''')
    st.markdown(
        r"""
        Here, we would **not** expect the conditional term $p(f|y)$ to be invariant. The amount of fertilizer a farmer likely used for a given yield depends on the overall economic conditions and practices of that environment.
        """
    )
    st.error("Thus, we expect to find that $p^{sf}(f|y) \\neq p^{if}(f|y)$.")

st.info(
    """
    **A Point of Clarification:** You might ask, "Is the farmer's choice truly independent of the crop's biology? A rational farmer considers the effect of the fertilizer when they make a decision."
    
    The PIM states that the underlying **modules** are autonomous.
    
    - The **Biological Mechanism** ($p(y|f)$) is a physical law. A crop's cellular response to nitrogen doesn't change if the farm gets bigger or smaller.
    - The **Farmer's Decision** ($p(f)$) is a behavioral mechanism. The farmer's brain uses a *belief* about the biological mechanism to make a choice, but the choice itself does not alter the biology.
    
    The principle holds because the modules are separate. The simulation works because the crop's biology is **invariant** even when the farmer's strategy changes between environments. For an example without this nuance, think of a scenrio without an intelligent agent such as $X$: Engine Size and $Y$: Fuel Efficiency.
    """,
    icon="💡"
)

st.markdown("---")
st.markdown("#### Conclusion from Theory")
st.markdown("By searching for the factorization that remains stable across the two different datasets, we can identify the causal direction. The simulation below performs exactly this test.")



st.divider()

# Simulation 1: Discovering the Cause through Invariance
st.header("Simulation: Discovering the Cause through Invariance")
st.markdown(
    """
    Let's test this principle. We'll simulate data from two different farming environments where the fertilizer usage patterns are different, but the crop biology is the same. We will then check which statistical model remains stable across both environments.
    """
)
with st.expander("Show the Ground Truth SCM"):
    st.latex(r'''
        \begin{aligned}
            F &:= N_F \quad (\text{Farmer's choice}) \\
            Y &:= 5 \cdot F + 20 + N_Y \quad (\text{Biological process})
        \end{aligned}
    ''')
    st.markdown("In our simulation, the distribution of $N_F$ will change between environments, but the function for $Y$ will not.")

environment = st.selectbox(
    "Choose a Farming Environment (this changes the distribution of the cause)",
    ("Small Farms", "Industrial Farms")
)

df = generate_data(environment)

col1_sim, col2_sim = st.columns(2)

with col1_sim:
    st.subheader("Test 1: Causal Direction ($F \\to Y$)")
    causal_eq = fit_and_get_equation(df, 'Fertilizer', 'Crop_Yield')
    st.markdown(f"**Fitted Model:** `{causal_eq}`")
    
    fig_causal = create_scatter_plot(df, 'Fertilizer', 'Crop_Yield', 'Yield vs. Fertilizer')
    st.plotly_chart(fig_causal, use_container_width=True)

with col2_sim:
    st.subheader("Test 2: Anti-Causal Direction ($Y \\to F$)")
    anticausal_eq = fit_and_get_equation(df, 'Crop_Yield', 'Fertilizer')
    st.markdown(f"**Fitted Model:** `{anticausal_eq}`")
    
    fig_anticausal = create_scatter_plot(df, 'Crop_Yield', 'Fertilizer', 'Fertilizer vs. Yield')
    st.plotly_chart(fig_anticausal, use_container_width=True)

st.success(
    """
    **Observation:** Toggle between the two environments. Notice that the equation for **Yield vs. Fertilizer** remains stable (always around `slope ≈ 5.0`). However, the equation for **Fertilizer vs. Yield** changes significantly.
    
    This stability, or **invariance**, is our evidence. It allows us to conclude that $F \\to Y$ is the correct causal direction.
    """
)

st.divider()

st.header("Theoretical Bridge: From Invariance to Independent Noise")
st.markdown(
    r"""
    The **invariance** we observed in the first simulation is a consequence of a deeper property, which is formalized in SCMs by the **independence of noise terms**.
    
    The Principle of Independent Mechanisms (PIM) states that the mechanism for the cause is independent of the mechanism for the effect. In an SCM, we formalize these mechanisms:
    -   **Mechanism for the Cause ($X$):** This is defined by the distribution of its noise term, $P(N_X)$. This gives us the marginal distribution $p(x)$.
    -   **Mechanism for the Effect ($Y$):** This is defined by the function $f_Y$ and the distribution of its noise term, $P(N_Y)$. This pair $(f_Y, P(N_Y))$ gives us the conditional distribution $p(y|x)$.

    Therefore, the statement "the two mechanisms are independent" translates directly into the mathematical assumption that their components are independent. Since $f_Y$ is a fixed, deterministic function, this simplifies to the noise distributions being independent:
    """
)
st.latex(r"N_X \perp\kern-5pt\perp N_Y")
st.markdown(
    """
    This is the foundational assumption. If we try to model the system in the wrong direction ($Y \\to X$), we are implicitly forcing a dependency between the inferred noise terms, which violates this core principle. The next simulation is designed to test for exactly this violation.
    """
)

st.divider()

st.subheader("How the Graphs Are Made: Inferring Noise with Residuals")
st.markdown(
    r"""
    We cannot observe the true noise terms $N_X$ and $N_Y$ directly from data. However, we can *estimate* them. The "inferred noise" shown in the plots below are the **residuals** from a linear regression model. Here is the step-by-step process:

    1.  **Hypothesize a Direction:** First, we assume a causal direction, for example, $X \to Y$.
    2.  **Model the Relationship:** We model this hypothesis with a linear equation: $Y = w \cdot X + \text{noise}$.
    3.  **Fit the Model:** We use Ordinary Least Squares (OLS) regression to find the best-fit line, which gives us an estimate for the coefficient, $\hat{w}$. This creates a prediction, $\hat{y}_i$, for each data point.
    4.  **Calculate the Inferred Noise (Residuals):** The residual is the part of $Y$ that is not explained by $X$. It's our empirical estimate of the true noise $N_Y$. For each data point $i$, the residual $r_i$ is:
    """
)
st.latex(r"r_i = y_i - \hat{y}_i = y_i - (\hat{w} \cdot x_i)")
st.markdown(
    """
    5.  **Check for Independence:** We then create a scatter plot of the assumed cause ($X$) against its inferred noise (the residuals, $r_i$). Loosely speaking, when they appear independent (a shapeless cloud), our hypothesis $X \\to Y$ is supported. If they show a clear pattern, the hypothesis is challenged. We then repeat the entire process for the anti-causal direction, $Y \\to X$.
    """
)


# Simulation 2 (The Formal Demo) 
st.markdown("---")
st.markdown("We use **Non-Gaussian** noise because dependencies are often much easier to see visually than with purely Gaussian data.")
with st.expander("Show the Ground Truth SCM"):
    st.latex(r'''
        \begin{aligned}
            N_X &\sim \text{Uniform}(-2, 2) \\
            N_Y &\sim \text{Exponential}(1) \\
            \\
            X &:= N_X \\
            Y &:= 2 \cdot X + N_Y
        \end{aligned}
    ''')


if st.button("Generate New Non-Gaussian Data"):
    st.session_state.lingam_df = generate_lingam_data()

if 'lingam_df' in st.session_state:
    df_lingam = st.session_state.lingam_df
    
    col1_lingam, col2_lingam = st.columns(2)
    
    with col1_lingam:
        st.subheader("Test 1: Causal Direction ($X \\to Y$)")
        residuals_causal = fit_and_get_residuals(df_lingam, 'X', 'Y')
        df_causal_analysis = pd.DataFrame({'Cause (X)': df_lingam['X'], 'Inferred Noise (Residuals)': residuals_causal})
        
        fig_causal = create_scatter_plot(df_causal_analysis, 'Cause (X)', 'Inferred Noise (Residuals)', 'Independence of Cause and Noise')
        st.plotly_chart(fig_causal, use_container_width=True)
        st.success("**Observation:** The inferred noise (residuals) forms a shapeless, random cloud. It appears statistically **independent** of the cause, as the PIM predicts for the correct causal direction.")

    with col2_lingam:
        st.subheader("Test 2: Anti-Causal Direction ($Y \\to X$)")
        residuals_anticausal = fit_and_get_residuals(df_lingam, 'Y', 'X')
        df_anticausal_analysis = pd.DataFrame({'Cause (Y)': df_lingam['Y'], 'Inferred Noise (Residuals)': residuals_anticausal})

        fig_anticausal = create_scatter_plot(df_anticausal_analysis, 'Cause (Y)', 'Inferred Noise (Residuals)', 'Dependence of Cause and Noise')
        st.plotly_chart(fig_anticausal, use_container_width=True)
        st.error("**Observation:** The inferred noise has a clear, sharp, parallelogram-like structure. It is statistically **dependent** on the cause. This violates the PIM, telling us this is the wrong causal direction.")

st.divider()

st.header("Conclusion & Key Takeaways")
st.success(
    """
    **The Principle of Independent Mechanisms is a powerful assumption for causal discovery.**
    
    You have now seen two ways this single principle can be used to infer causal direction from purely observational data:
    
    1.  **Finding Invariance:** By comparing data from different environments, we can identify the causal direction as the one whose conditional distribution $p(\\text{effect}|\\text{cause})$ remains stable even when the distribution of the cause $p(\\text{cause})$ changes.
    
    2.  **Checking for Noise Independence:** For a single dataset, we can test both causal directions. The correct direction is the one where the inferred noise is statistically independent of the cause.
    
    These methods are the foundation of many modern causal discovery algorithms.
    """
)