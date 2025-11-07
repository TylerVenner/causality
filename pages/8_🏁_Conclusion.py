import streamlit as st

st.set_page_config(
    page_title="Final Thoughts",
    page_icon="🏁",
    layout="wide"
)

st.title("🏁 Final Thoughts")

st.markdown(
    """
    I started this project as a statistics student with what felt like a
    simple question: **"What *is* causality?"**
    
    I felt I was missing the language to talk about *cause and effect*.
    I wanted to know how we could get from "A and B are related" to
    "A *causes* B" or at least think about it in a more principled way.
    
    My primary guide for this journey has been the textbook
    *Elements of Causal Inference* by Jonas Peters, Dominik Janzing, 
    and Bernhard Schölkopf, which provided the formal
    grounding for this entire app.

    This Streamlit app is the highlights of my two-month journey to find an
    answer. What I discovered was a challenging and exciting field of research 
    at the intersection of statistics, computer science, and philosophy.
    
    We've traveled from the basic idea of an intervention, up Pearl's
    "Ladder of Causation" to counterfactuals, and all the way to
    discovery algorithms like PC and FCI.
    """
)

st.divider()

st.header("Limitations and Assumptions")
st.markdown(
    """
    If I've learned one thing, it's that the tools we've explored
    are just the first step. They are incredibly powerful, but they
    are not magic. They all come with real-world limitations.
    """
)

st.markdown(
    r"""
    **The Limitations We've Built On:**
    
    * **The Causal Markov Property:** We've implicitly assumed that our
        SCM's structure (the graph) perfectly explains all the
        independencies in the data. This is the fundamental link
        ($Graph \implies Independencies$) that lets us use d-separation.
    
    * **The Faithfulness Assumption:** We've also assumed the
        *opposite*: that *every* independence in the data comes
        from the graph structure, and not from "accidental"
        cancellations. If $A$ causes $C$ through two paths that
        perfectly cancel each other out, the algorithm will (incorrectly)
        find $A \perp C$ and remove the edge.
    
    * **Causal Sufficiency (PC):** The PC algorithm *requires*
        this assumption—that we've measured *all*
        confounders. As we saw, this is almost never
        true in messy, real-world observational data, leading to
        confidently wrong answers.
    
    * **FCI as a Tradeoff:** The FCI algorithm is a brilliant
        solution to this problem, but it trades precision for honesty.
        It gives us a complex **PAG** that correctly tells us
        "I don't know" (using $\leftrightarrow$ or $\circ- \circ$),
        which is correct, but harder to use.
    
    * **Identifiability:** We saw that some things, like the
        $H_1$ variable, are fundamentally **unidentifiable**
        from observational data alone. The algorithms can only
        see the "shadows" that hidden variables cast.
    """
)

st.header("An Active Field of Research")
st.markdown(
    r"""
    Great minds are
    currently tackling questions that go far beyond the scope
    of my introductory app:
    
    * **Causal Representation Learning (CRL):** We assumed we had
        a nice table of variables like $A, B, C$. But what if our
        data is raw images, audio, or text? CRL asks: how can we
        learn the *causal variables themselves* from high-dimensional
        data?
    
    * **Non-stationarity & Heterogeneity:** We assumed our data's
        rules were stable (i.i.d.). But what about systems whose
        causal laws *change* over time (non-stationarity) or
        differ across groups (heterogeneity)?
    
    * **Cycles & Time-Series:** Our graphs were all
        **D**irected **A**cyclic **G**raphs (DAGs). What about
        systems with feedback loops (e.g., in economics or
        biology)?
    
    There is much more under active research, and I think it's very exciting!
    """
)

st.divider()

st.header("My Hope")
st.markdown(
    """
    I built this app to re-trace my own steps and spark interest in other people.
    What we've covered here, a few key concepts and two algorithms, is
    truly just the first layer.
    
    My hope is that this demo has sparked the same curiosity in you.
    I hope it's made you skeptical of "correlation equals causation"
    in a new, more structured way, and given you a glimpse of the
    tools we have to do better.

    Thank you.
    """
)