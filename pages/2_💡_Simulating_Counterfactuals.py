import streamlit as st
from src.simulations.counterfactual_sim import solve_for_nb, calculate_counterfactual_outcome

st.set_page_config(
    page_title="Simulation 2: Counterfactuals",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Simulation 2: Counterfactuals")
st.markdown(
    """
    We now move to: **counterfactuals**. 
    While interventions ask "what if we do X?", counterfactuals ask "what if we *had done* X differently?", given an outcome that we have already observed.
    
    This simulation interactively walks through the formal three-step process of answering such a "what if" question.
    """
)
st.subheader("The Causal Hierarchy")
st.markdown(
    """
    Before diving into the simulation, it's helpful to understand where counterfactuals sit within the broader framework of causal reasoning. Judea Pearl organized causal questions into a three-level hierarchy.
    
    1.  **Level 1: Association (Seeing)**
        - **Question:** "What if I see...?" — *What does a symptom tell me about a disease?*
        - **Math:** Deals with standard conditional probabilities, $P(Y|X=x)$. This is most standard machine learning.
    
    2.  **Level 2: Intervention (Doing)**
        - **Question:** "What if I do...?" — *What happens if I administer this drug?*
        - **Math:** Deals with interventional probabilities, $P(Y|\\text{do}(X=x))$. This requires a causal model to predict the effects of actions.
    
    3.  **Level 3: Counterfactuals (Imagining)**
        - **Question:** "What if I had done...?" — *What if the patient who died had been given a different drug?*
        - **Math:** Deals with counterfactual probabilities, $P(Y_x|X=x', Y=y')$. This is the most powerful level, allowing us to reason about specific events in alternate realities.
    
    This simulation focuses on Level 3.
    """
)
st.divider()

# --- Section 1: The Ground Truth SCM ---
st.header("The Ground Truth SCM: The 'Eye Disease' Model")
st.markdown(
    """
    Our scenario involves a **Treatment ($T$)** for an eye disease, which results in an **Outcome ($B$)**. The result is complicated by a hidden, rare genetic **Condition ($N_B$)**.
    - **$T=1$**: Treatment given, **$T=0$**: No treatment.
    - **$B=1$**: Patient goes blind, **$B=0$**: Patient is cured.
    - **$N_B=1$**: Patient has the rare condition (1% of population), **$N_B=0$**: Patient does not (99% of population).
    
    The logic of this world is captured by the following SCM, where $N_T$ represents the doctor's decision process:
    """
)
with st.container(border=True):
    st.latex(r'''
        \begin{aligned}
            N_B &\sim \text{Bernoulli}(0.01) \\
            N_T &\sim \text{Some Distribution} \\
            \\
            T &:= N_T \\
            B &:= T \cdot N_B + (1-T) \cdot (1-N_B)
        \end{aligned}
    ''')

st.divider()

# --- Section 2: Interactive Counterfactual Analysis ---
st.header("Interactive Counterfactual Analysis")

# Define the factual scenario
FACTUAL_T = 1
FACTUAL_B = 1

st.info(
    f"""
    **The Factual Scenario:** A specific patient comes to the hospital. We observe the following facts:
    - The doctor administered the treatment (**$T={FACTUAL_T}$**).
    - The patient went blind (**$B={FACTUAL_B}$**).
    
    **The Counterfactual Question:** What would have happened to this *specific* patient if the doctor had **not** administered the treatment?
    """
)

st.subheader("The Three-Step Process")

# Step 1: Abduction
with st.expander("#### Step 1: Abduction (The Detective Step)"):
    st.markdown(
        f"""
        First, we use the observed facts ($T={FACTUAL_T}, B={FACTUAL_B}$) to deduce the value of the unobserved exogenous variable, $N_B$, for this individual. We plug the facts into our SCM's equation for $B$:
        """
    )
    st.latex(f"B = T \\cdot N_B + (1-T) \\cdot (1-N_B)")
    st.latex(f"{FACTUAL_B} = {FACTUAL_T} \\cdot N_B + (1-{FACTUAL_T}) \\cdot (1-N_B)")
    st.latex(f"{FACTUAL_B} = 1 \\cdot N_B + 0 \\cdot (1-N_B) \implies N_B = {FACTUAL_B}")
    
    deduced_nb = solve_for_nb(T=FACTUAL_T, B=FACTUAL_B)
    st.success(f"**Conclusion:** For this specific patient, the hidden condition **$N_B$ must have been {deduced_nb}**.")

# Step 2: Action
with st.expander("#### Step 2: Action (The Time-Traveler Step)"):
    st.markdown(
        """
        Next, we take our knowledge about this specific patient (i.e., we fix $N_B$ to its deduced value) and apply our hypothetical action. We use the $do$-operator to replace the original action with our counterfactual one.
        """
    )
    COUNTERFACTUAL_T = 0
    st.latex(fr"\text{{Counterfactual Action: }} do(T := {COUNTERFACTUAL_T})")
    st.markdown(
        f"""
        The original world for this patient was described by the SCM with $N_B={deduced_nb}$. 
        Our counterfactual world is described by a modified SCM where we have forced $T$ to be ${COUNTERFACTUAL_T}$.
        """
    )

# Step 3: Prediction
with st.expander("#### Step 3: Prediction (The Prophet Step)"):
    st.markdown(
        f"""
        Finally, we compute the outcome in this new, hypothetical world using the modified SCM. We use the deduced $N_B={deduced_nb}$ and the counterfactual action $T={COUNTERFACTUAL_T}$.
        """
    )
    counterfactual_b = calculate_counterfactual_outcome(deduced_nb=deduced_nb, counterfactual_T=COUNTERFACTUAL_T)
    st.latex(f"B_{{cf}} = T_{{cf}} \\cdot N_B + (1-T_{{cf}}) \\cdot (1-N_B)")
    st.latex(f"B_{{cf}} = {COUNTERFACTUAL_T} \\cdot {deduced_nb} + (1-{COUNTERFACTUAL_T}) \\cdot (1-{deduced_nb})")
    st.latex(f"B_{{cf}} = 0 + 1 \\cdot {1-deduced_nb} = {counterfactual_b}")

st.image("assets/counterfactual_flow.png")


outcome_text = "Cured (B=0)" if counterfactual_b == 0 else "Blind (B=1)"
st.markdown(f"**Conclusion:** Thus, had this patient *not* been treated, they would have been **{outcome_text}**.")

st.markdown(
    """
    This result seems paradoxical. The doctor's decision was based on the fact that the treatment has a 99% success rate for the general population ($P(B=0|do(T=1))=0.99$). Indeed, this was the optimal decision. However, for this *one specific patient*, the treatment was the wrong choice. 
    """
)

st.header("Formal Distinction: Interventions vs. Counterfactuals")
st.markdown(
    """
    The simulation highlights a crucial difference in how we use the SCM. Let's formalize this.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Interventions (Population Level)")
    st.markdown(
        r"""
        An intervention asks about the **average effect** over a population. The calculation involves two steps:
        
        1.  **Modify:** Create a new, "mutilated" SCM, $\mathfrak{C}_{do(T:=t)}$, by replacing the equation for $T$.
        2.  **Average:** Calculate the expected outcome by averaging over the entire distribution of all possible noise values, $P(N_B)$.
        
        This gives us a population-level statistic, like the 99% success rate of the drug in our example:
        """
    )
    st.latex(r"E[B | do(T:=1)] = 0.01")
    
with col2:
    st.subheader("Counterfactuals (Individual Level)")
    st.markdown(
        r"""
        A counterfactual asks about a **specific individual** for whom we have factual observations. The calculation involves three steps:
        
        1.  **Abduction:** Use the observed facts (e.g., $T=1, B=1$) to pinpoint the **specific value** of the noise for that individual (e.g., $N_B=1$).
        2.  **Modify:** Create the mutilated SCM, $\mathfrak{C}_{do(T:=t')}$.
        3.  **Predict:** Calculate the outcome in the new model using the **specific noise value** we found in step 1.
        
        This gives us an individual-level prediction, which can be different from the population average:
        """
    )
    st.latex(r"B_{do(T:=0)}(N_B=1) = 0")

st.markdown(
    """
    **In short:** Interventions average over all possibilities, while counterfactuals narrow down to one specific possibility before predicting the outcome of a hypothetical action.
    """
)


st.header("Are Counterfactuals Scientific? A Falsifiable Claim")
st.markdown(
    """
    A fair question is whether this is all just an untestable story. The answer is no. A counterfactual is a testable hypothesis about your SCM. If the hidden noise variable ($N_B$) is measurable in principle, we can check if our model's deductions match reality.
    """
)
st.info(
    f"""
    **Scenario:** Our SCM-based reasoning concluded that for our patient, their hidden condition **must have been $N_B=1$**. Imagine a genetic test can measure this directly, and the late lab results have just arrived.
    
    Click a button below to reveal a possible test result and see what it means for our model.
    """
)

if 'lab_result' not in st.session_state:
    st.session_state.lab_result = None

col_btn1, col_btn2 = st.columns(2)
if col_btn1.button("Reveal Result: Test shows **$N_B = 1$**", use_container_width=True):
    st.session_state.lab_result = 'confirmed'

if col_btn2.button("Reveal Result: Test shows **$N_B = 0$**", use_container_width=True):
    st.session_state.lab_result = 'falsified'

# Display the outcome based on which button was pressed
if st.session_state.lab_result == 'confirmed':
    st.success(
        """
        **Outcome: Model Confirmed**
        
        The physical evidence ($N_B=1$) perfectly matches our model's deduction. This gives us confidence that our SCM accurately represents the real-world mechanism, and validates our counterfactual conclusion.
        """
    )
elif st.session_state.lab_result == 'falsified':
    st.error(
        """
        **Outcome: Model Falsified!**
        
        The physical evidence ($N_B=0$) directly **contradicts** our model's deduction. Our SCM predicted that a patient with $N_B=0$ who receives treatment ($T=1$) *should have been cured* ($B=0$). But this patient went blind.
        
        This contradiction proves that our SCM is **wrong**. The real world works differently than our model assumed, and we must revise it. This is how counterfactuals can be falsifiable.
        """
    )

st.header("Why Couldn't a Standard Machine Learning Model Do This?")
st.markdown(
    """
    It's worth asking how a traditional machine learning model would handle this scenario.
    
    - A classifier trained on the hospital's data would learn the observational distribution, $P(B|T)$. It would learn that treatment is overwhelmingly associated with being cured (99% of the time).
    - If shown our patient's case ($T=1$), it would predict the most likely outcome: **Cured ($B=0$)**.
    - When the patient was instead observed to be blind ($B=1$), the ML model would classify this as a rare error or unexplainable noise. It has no mechanism to **explain** *why* this specific patient had a different outcome.
    
    The SCM, by contrast, provides a powerful explanation through abduction: the model wasn't wrong, the patient simply belonged to a rare subgroup ($N_B=1$). This ability to reason about specific, individual-level causes for observed outcomes is a key advantage of the causal approach.
    """
)

st.success(
    """
    **Final Takeaway:**
    
    You've now seen the full power of the Causal Hierarchy. Counterfactuals allow us to move beyond population-level intervention and conduct retrospective analysis on specific events. This provides a formal framework for asking "what if?".
    """
)