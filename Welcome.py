import streamlit as st

st.set_page_config(
    page_title="Causal Inference",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This app explores Causal Inference."
    }
)

st.title("Welcome to my Interactive Causal Inference App!")

st.markdown(
    """
    This app is an interactive application designed to introduce 
    **Causal Inference**. We'll explore everything from the foundational concepts to
    advanced algorithms for causal discovery.
    """
)

st.divider()

st.subheader("About This Project")
st.markdown(
    """
    I'm a statistics student and data scientist who wanted to look more closely at the 
    phrase "correlation does not imply causation." It started as simple curiosity but quickly
    grew to a something more.
    
    Here, I've gathered some of the highlights from that exploration to built this app, 
    hoping to create an interactive, hands-on
    guide for other students and other curious people.
    
    This app follows my own two-month journey from the basic question
    "What is causality?" to implementing discovery algorithms and understanding
    assumptions.
    My primary guide for this journey has been the textbook
    *Elements of Causal Inference* by Jonas Peters, Dominik Janzing, 
    and Bernhard Schölkopf.
    
    ---
    
    **Find the project on GitHub:**
    * **[Repository Link](https://github.com/TylerVenner/causality)** (Feel free to fork or star!)
    * **[Report a Bug](https://github.com/TylerVenner/causality/issues)** (Let me know if you find issues.)
    
    **Connect with me:**
    * **[My LinkedIn Profile](www.linkedin.com/in/tylervenner)** (Let's connect!)
    """
)

st.subheader("The Structure of this Project")
st.markdown(
    """
    Click any chapter below to begin, or use the sidebar to navigate.
    Each page builds on the last:
    """
)

st.page_link("pages/0_📖_Introduction.py", 
            label="**0. Introduction:** Start here!", 
            icon="📖")
st.page_link("pages/1_🔬_Asymmetry_of_Interventions.py", 
            label="**1. Asymmetry of Interventions:** *Seeing* vs. *Doing*", 
            icon="🔬")
st.page_link("pages/2_💡_Simulating_Counterfactuals.py", 
            label="**2. Simulating Counterfactuals:** Asking 'What if?'", 
            icon="💡")
st.page_link("pages/3_🧠_Independence_of_Mechanism.py", 
            label="**3. Independence of Mechanism:** The core assumption", 
            icon="🧠")
st.page_link("pages/4_🔗_Confounding_vs_Mediation.py", 
            label="**4. Confounding vs. Mediation:** Untangling paths", 
            icon="🔗")
st.page_link("pages/5_🗺️_The_Causal_Markov_Property.py", 
            label="**5. The Causal Markov Property:** Graphs and probabilities", 
            icon="🗺️")
st.page_link("pages/6_🧭_PC_Algorithm.py", 
            label="**6. PC Algorithm:** Our first discovery tool", 
            icon="🧭")
st.page_link("pages/7_👻_Hidden_Confounding_and_FCI.py", 
            label="**7. Hidden Confounding & FCI:** When assumptions fail", 
            icon="👻")
st.page_link("pages/8_🏁_Conclusion.py", 
            label="**8. Conclusion:** My final thoughts", 
            icon="🏁")

st.sidebar.success("Select a demo above to begin.")