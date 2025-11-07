import streamlit as st
import statsmodels.formula.api as smf

from src.simulations.confounding_vs_mediation_sim import generate_confounding_data, generate_mediation_data
from src.plotting.charts import create_scatter_plot, create_colored_scatter_plot

st.set_page_config(
    page_title="Confounding vs. Mediation",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Confounding vs. Mediation")
st.markdown(
    """
    One of the most practical questions in data analysis is, "Should I control for variable Z?" The answer, as we'll see, is **it depends on the causal structure**. 
    
    Adjusting for a variable can either eliminate a spurious correlation (good!) or block a real causal path (bad!). This interactive simulation demonstrates these two critical scenarios.
    """
)

st.divider()

# SCENARIO 1: CONFOUNDING 
st.header("Scenario 1: The Confounder (Adjusting is Necessary)")
st.markdown(
    """
    **Story:** A company observes a strong correlation between its Ad Spend ($X$) and Sales ($Y$). The hidden variable is the **Holiday Season ($Z$)**, which is a **common cause** of both higher ad spending and higher sales.
    """
)

# Causal graph for confounding
st.graphviz_chart("""digraph { rankdir=LR; Z [label="Holiday Season (Z)"]; X [label="Ad Spend (X)"]; Y [label="Sales (Y)"]; Z -> X; Z -> Y; X -> Y [style=dashed, label="small true effect"];}""")

with st.container(border=True):
    st.latex(r'''
        \begin{aligned}
            N_Z &\sim \text{Bernoulli}(0.2) \\
            N_X &\sim \mathcal{N}(5, 2^2), \quad N_Y \sim \mathcal{N}(50, 5^2) \\
            \\
            Z &:= N_Z \\
            X &:= 20 \cdot Z + N_X \\
            Y &:= 50 \cdot Z + 2 \cdot X + N_Y
        \end{aligned}
    ''')
    st.markdown("Note the true causal effect of $X$ on $Y$ has a coefficient of **2.0**.")

# Generate confounding data
df_confounding = generate_confounding_data()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Visual Evidence")
    st.markdown("The scatter plot shows the observational data. The colors reveal the influence of the confounder, creating two distinct groups of data.")
    fig_colored = create_colored_scatter_plot(df_confounding, 'Ad_Spend', 'Sales', 'Holiday_Season', 'Sales vs. Ad Spend (Colored by Confounder)')
    st.plotly_chart(fig_colored, use_container_width=True)

with col2:
    st.subheader("Statistical Analysis")
    st.markdown("Use the toggle below to see how adjusting for the confounder changes the result.")
    
    adjust_confounder = st.checkbox("Adjust for Holiday Season (Z)", value=False)
    
    if adjust_confounder:
        st.markdown("##### Corrected Analysis (Adjusted)")
        model_adjusted = smf.ols('Sales ~ Ad_Spend + Holiday_Season', data=df_confounding).fit()
        p = model_adjusted.params
        st.code(f"Sales ≈ {p['Ad_Spend']:.2f} * Ad_Spend + {p['Holiday_Season']:.2f} * Holiday_Season + {p['Intercept']:.2f}")
        st.success(f"**Conclusion:** After adjusting, the coefficient for Ad Spend is **{p['Ad_Spend']:.2f}**. This is very close to the true causal effect of **2.0**.")
    else:
        st.markdown("##### Naive Analysis (Unadjusted)")
        model_naive = smf.ols('Sales ~ Ad_Spend', data=df_confounding).fit()
        p = model_naive.params
        st.code(f"Sales ≈ {p['Ad_Spend']:.2f} * Ad_Spend + {p['Intercept']:.2f}")
        st.warning(f"**Conclusion:** The naive analysis shows a large coefficient of **{p['Ad_Spend']:.2f}**, which is misleadingly inflated by the confounder.")
        
st.divider()

# SCENARIO 2: MEDIATION 
st.header("Scenario 2: The Mediator (Adjusting is a Mistake)")
st.markdown(
    """
    **Story:** In a different company, Ad Spend causes more **Website Clicks ($Z$)**, which in turn leads to more Sales ($Y$). Clicks are a **mediator** on the causal pathway.
    """
)

st.graphviz_chart("""digraph { rankdir=LR; X [label="Ad Spend (X)"]; Z [label="Website Clicks (Z)"]; Y [label="Sales (Y)"]; X -> Z -> Y;}""")

with st.container(border=True):
    st.latex(r'''
        \begin{aligned}
            N_X &\sim \text{Uniform}(1, 10) \\
            N_Z &\sim \mathcal{N}(10, 5^2), \quad N_Y \sim \mathcal{N}(20, 10^2) \\
            \\
            X &:= N_X \\
            Z &:= 10 \cdot X + N_Z \\
            Y &:= 5 \cdot Z + N_Y
        \end{aligned}
    ''')

# Generate mediation data
df_mediation = generate_mediation_data()

col3, col4 = st.columns([1, 1])
with col3:
    st.subheader("Visual Evidence")
    st.markdown("The scatter plot shows the strong *total effect* of Ad Spend on Sales. The entire effect flows through the mediator, Website Clicks.")
    fig_total = create_scatter_plot(df_mediation, 'Ad_Spend', 'Sales', 'Sales vs. Ad Spend (Total Effect)')
    st.plotly_chart(fig_total, use_container_width=True)

with col4:
    st.subheader("Statistical Analysis")
    st.markdown("Use the toggle to see what happens when you mistakenly adjust for the mediator.")
    
    adjust_mediator = st.checkbox("Adjust for Website Clicks (Z)", value=False)
    
    if adjust_mediator:
        st.markdown("##### Incorrect Analysis (Adjusted)")
        model_blocked = smf.ols('Sales ~ Ad_Spend + Website_Clicks', data=df_mediation).fit()
        p = model_blocked.params
        st.code(f"Sales ≈ {p['Ad_Spend']:.2f} * Ad_Spend + {p['Website_Clicks']:.2f} * Website_Clicks + {p['Intercept']:.2f}")
        st.error(f"**Conclusion:** After adjusting, the coefficient for Ad Spend is **{p['Ad_Spend']:.2f}**. This is wrong! By controlling for the mediator, we blocked the causal path, making it seem like ads have no effect.")
    else:
        st.markdown("##### Correct Analysis (Total Effect)")
        model_total_effect = smf.ols('Sales ~ Ad_Spend', data=df_mediation).fit()
        p = model_total_effect.params
        st.code(f"Sales ≈ {p['Ad_Spend']:.2f} * Ad_Spend + {p['Intercept']:.2f}")
        st.success(f"**Conclusion:** The unadjusted model correctly shows the strong, positive **total effect** of Ad Spend on Sales, with a coefficient of **{p['Ad_Spend']:.2f}**.")

st.divider()

st.header("The Rules of Adjustment")
st.markdown("Based on these simulations, we can derive two fundamental rules for when to control for a third variable, Z.")

rule1, rule2 = st.columns(2)
with rule1:
    with st.container(border=True):
        st.subheader("Rule 1: Adjust for Confounders")
        st.graphviz_chart("""digraph { rankdir=LR; Z [label="Confounder"]; X; Y; Z -> X; Z -> Y;}""")
        st.success(
            """
            **You SHOULD adjust for a common cause (confounder).** This blocks the non-causal "back-door" path from X to Y, removing spurious correlation and isolating the true causal effect of X on Y.
            """
        )

with rule2:
    with st.container(border=True):
        st.subheader("Rule 2: Do NOT Adjust for Mediators")
        st.graphviz_chart("""digraph { rankdir=LR; Z [label="Mediator"]; X -> Z -> Y;}""")
        st.error(
            """
            **You SHOULD NOT adjust for a variable on the causal pathway (a mediator)** if you want to estimate the total effect of X on Y. This blocks the very mechanism you are trying to measure, leading to an incorrect estimate.
            """
        )