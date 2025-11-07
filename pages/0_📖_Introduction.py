# pages/0_📖_Why_Causal_Inference.py

import streamlit as st

st.set_page_config(
    page_title="Why Causal Inference?",
    page_icon="📖",
    layout="wide"
)

st.title("📖 From Association to Causation")

st.markdown(
    """
    This page introduces the fundamental difference between two paradigms: **statistical learning**, which models associations (correlations), and **causal inference**, which provides the tools to predict the outcomes of actions. We'll explore why this distinction is essential for reliable decision-making.
    """
)

st.divider()

# --- Section 1: Statistical Learning ---
st.header("The Statistical Learning Paradigm: A Single Distribution")
st.markdown(
    r"""
    The framework for traditional statistical learning is based on a single, fixed data-generating distribution.
    
    - **The Setup:** We have a pair of random variables $(X, Y)$ with a joint probability distribution $P(X, Y)$. We are given a set of observational data $\{(x_i, y_i)\}_{i=1}^N$ drawn i.i.d. from this distribution.
    
    - **The Goal:** The primary goal is **prediction**. We want to find a function $\hat{f}(x)$ that is a good approximation of the true regression function $f(x) = E[Y|X=x]$. This function answers the question:
    
    > *"Given we **observe** a new input vector $X=x$, what is our best prediction for the value of $Y$?"*
    """
)
st.markdown(
    r"""
    - **The Mathematics:** All inferences are about properties of the joint distribution $P(X, Y)$ or the conditional distribution $P(Y|X)$. The methods are designed to model the **associations** or **correlations** present in the data. The entire theoretical foundation (bias-variance tradeoff, etc.) is built upon the assumption of sampling from this single, fixed distribution.
    """
)

st.divider()

# --- Section 2: Causal Inference ---
st.header("The Causal Inference Paradigm: A Family of Distributions")
st.markdown(
    """
    Causal inference addresses a fundamentally different and more difficult question. It is not about observing, but about **intervening**. The goal is to predict what would happen if we were to *change* the system.

    - **The Setup:** A causal model is not a single distribution, but rather a structure that implies a whole **family of potential distributions**, one for each possible intervention. This structure describes how variables influence one another.
    
    - **The Goal:** The goal is to answer a counterfactual or interventional question:
    
    > "If we were to **intervene** and set the value of $X$ to $x$, what would be the resulting value of $Y$?"

    > *"If we launch this ad campaign, what will happen to sales?"* \n
    > *"If we implement this public policy, what will be the effect on public health?"* \n
    > *"If we prescribe this drug, what is the patient\'s expected outcome?"*
    """
)
st.markdown(
    r"""
    **The Mathematics:** This conceptual difference is formalized mathematically using the $do()$-operator. Causal inference is concerned with estimating quantities like $E[Y|\text{do}(X=x)]$.

    - The **statistical quantity** $E[Y|X=x]$ is the *conditional expectation*. It is a property of a single distribution $P(X,Y)$. It describes a passive observation. It looks at all the values of $Y$ whenever it just so happens that $X$ falls at the value of $x$.
    
    - The **causal quantity** $E[Y|\text{do}(X=x)]$ is the *interventional expectation*. It describes an active intervention. An intervention on $X$ fundamentally changes the system, creating a new, different distribution for the remaining variables. It looks at all the values of $Y$ while forcing all values of $X$ to be $x$. 
    """
)

st.subheader("Why are these different?")
st.markdown(
    r"""
    The classic example is confounding. Let $X$ be scarf sales and $Y$ be the number of people who get hyperthermia. In the observational distribution, $E[Y|X=\text{high}]$ is high because a third variable, temperature ($Z$), causes both. If we intervene to set sales of scarfs high (e.g., by giving a drastic discount on price during the summer), we do not expect number of people with hyperthermia to increase. Thus, the two quantities are not equal:
    """
)
st.latex(r"E[Y|\text{do}(X=\text{high})] \neq E[Y|X=\text{high}]")
st.markdown(
    """
    The statistical model captures the association, but the causal model is needed to correctly predict the effect of an action.
    """
)

st.header("Example: Advertising and Sales")
col_text, col_fig = st.columns([1, 1])

with col_text:
    st.markdown(
        """
        This example provides a contrast between a **statistical prediction** (what we expect to see) and a **causal prediction** (what we expect to happen after an intervention). We'll see how two variables can be statistically identical in an observational setting, yet have completely different causal roles.
        
        Let $Y$ be **Product Sales** and $X$ be **Advertising Spend**. From observational data, we see that $X$ and $Y$ are strongly correlated, as shown in the plot. A statistical model, such as a linear regression model, would work very nicely here.
        
        But what can we tell about the causal relationship about $X$ and $Y$? Does $X \\to Y$ (you can read as $X \\to Y$ as $X$ causes $Y$) or $Y \\to X$? Or something else? If you are a part of this business, knowing the correlation would not be enough. You need to know the causal structure underlying $X$ and $Y$ to make decisions such as "We should spend more on advertisment".
        """
    )
with col_fig:
    st.image(
        "assets/observational_data.png",
        caption="Observational data showing a strong positive correlation between ad spend and sales."
    )

st.subheader("Two Scenarios: Identical Statistics, Different Causal Realities")
st.markdown("The following two causal stories could have generated the *exact same* observational data. A data analyst with infinite data could not tell them apart without performing an experiment.")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Scenario A: Direct Causation")
    st.markdown("- **Causal Story:** Advertising spend ($X_{Ad}$) has a direct causal influence on product sales ($Y$). More spending leads to more sales.")
    st.markdown("- **Causal Graph:** $X_{Ad} \\to Y$")
    st.markdown("- **Statistical Prediction:** If we observe high ad spend, we predict high sales.")
    st.markdown("- **Causal Prediction:** If we **intervene** and cut the ad budget ($do(X_{Ad}=0)$), we expect sales to drop significantly. The change in $X_{Ad}$ directly impacts the mechanism that generates $Y$.")


with col_b:
    st.markdown("#### Scenario B: Common Cause (Confounding)")
    st.markdown("- **Causal Story:** Ad spend ($X_{Ad}$) has no direct effect on sales. Instead, an unmeasured factor $Z$ (e.g., the **Holiday Season**) causes both a rise in ad spend *and* a rise in sales. (Maybe ads are simply more expensive during holidays!)")
    st.markdown("- **Causal Graph:** $X_{Ad} \\leftarrow Z \\to Y$")
    st.markdown("- **Statistical Prediction:** If we observe high ad spend, we can infer it's likely the holiday season, so we predict high sales. The correlation is just as strong as in Scenario A.")
    st.markdown("- **Causal Prediction:** If we **intervene** and cut the ad budget ($do(X_{Ad}=0)$), we do not cancel the holiday season. The mechanism for sales, which depends on $Z$, is **completely unchanged**. We predict that sales will remain high regardless of our action on advertising.")

st.image("assets/ad_causal_graphs.png", caption="Two different causal structures that can produce identical observational data but make opposite predictions under intervention.", width='stretch')

st.subheader("The Need for Causal Models")
st.markdown(
    """
    - **Observational Indistinguishability:** From purely observational data, we cannot distinguish between Scenario A and Scenario B. A standard predictive model ($P(Y|X)$) would perform equally well on both as it only sees correlation.
    
    - **Interventional Divergence:** The two scenarios make completely different predictions about the outcome of an action (cutting the ad budget). One predicts a major drop in sales, the other predicts no change at all.
    
    - **The Role of Causal Knowledge:** To answer the question "What will happen if we change the ad budget?", we **must** have a causal model. The statistical correlation alone is insufficient and can be dangerously misleading for decision-making.
    """
)

# --- Section 3: Formal Causal Theory ---
st.header("Formalizing the Causal Story: Structural Causal Models (SCMs)")
st.markdown(
    """
    We use **Structural Causal Models (SCMs)** as our core mathematical framework. An SCM describes the data-generating process based on causal mechanisms.
    """
)
st.subheader("The Bivariate Case: Cause and Effect")
st.markdown(
    """
    Let's formalize the simplest causal graph, $C \\to E$ (Cause $\\to$ Effect). The SCM consists of two structural assignments:
    """
)
st.latex(r'''
    \begin{aligned}
        C &:= N_C \\
        E &:= f_E(C, N_E)
    \end{aligned}
''')
st.markdown(
    r"""
    - The notation $:=$ represents a **causal assignment**, not a mathematical equality. It means the value of the variable on the left is determined by the mechanism (the function) on the right.
    - The first assignment, $C := N_C$, states that the cause $C$ is determined by factors (noise) outside the model.
    - The second assignment, $E := f_E(C, N_E)$, states that the effect $E$ is determined by a function of its cause $C$ and its own independent noise $N_E$.
    - The core assumption, **$N_C \perp\kern-5pt\perp N_E$**, means that the unmodeled factors influencing $C$ are independent of the unmodeled factors influencing $E$. More on this later.
    
    If you are given the functions and noise distributions, you can perfectly simulate the system, which in turn generates the joint distribution $P(C, E)$ that we observe.
    """
)

st.markdown(
    r"""
    The SCM framework allows us to model what happens when we actively *change* a system. Let $\mathfrak{C}$ be a SCM. An expression like $P_E^{\mathfrak{C}}(e | C=c)$ refers to the observational distribution (what is the probability of effect $E$ in the sub-population where we *see* cause $C=c$?).
    
    In contrast, an expression like $P_E^{\mathfrak{C};\text{do}(C:=c)}$ refers to the interventional distribution (what is the probability of effect $E$ if we *force* the cause $C$ to be $c$ for everyone?). 
    """
)

st.subheader('The do-operator as a "Model Surgery"')
st.markdown(
    """
    When we perform a hard intervention like $do(C:=c)$, we create a new, modified SCM.
    
    Imagine our original SCM, $\mathfrak{C}$, is:
    """
)
st.latex(r'''
    \mathfrak{C}: \begin{cases}
        C &:= N_C \\
        E &:= f_E(C, N_E)
    \end{cases}
''')
st.markdown(
    """
    The intervention $do(C:=c)$ modifies the model by:
    1.  Finding the equation for $C$, which is $C := N_C$.
    2.  **Deleting** this equation from the model.
    3.  **Replacing** it with the new assignment, $C := c$.

    The resulting "mutilated" SCM, $\mathfrak{C'}$ is:
    """
)
st.latex(r'''
    \mathfrak{C'}: \begin{cases}
        C &:= c \\
        E &:= f_E(c, N_E)
    \end{cases}
''')

st.markdown(
    """
    The SCM, $\mathfrak{C'}$, describes a new reality where the natural mechanism for $C$ no longer applies, but the mechanism for $E$ remains exactly as it was.
    """
)


st.divider()

st.success(
    """
    **Now, navigate to the simulations in the sidebar to see these principles in action!**
    """
)