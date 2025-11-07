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

st.header("The 'Big Leap': Why This Matters for AI")
st.markdown(
    """
    My biggest takeaway is the potential for this field to drive the
    next big leaps in Artificial Intelligence.
    
    Most modern machine learning is based on statistical association. 
    It's brilliant at learning $P(Y|X)$
    from a static dataset, but this model is brittle. When the 
    real-world data distribution shifts, the model's learned 
    correlations may no longer hold.
    
    Humans, on the other hand, act upon what we believe are true
    causal structures in addition to statistical associations.
    """
)

st.subheader("The Process of Adaptation is the Prize")
st.markdown(
    """
    As humans, we rarely start with a "correct" causal model. As kids
    (and even as experts), we operate on flawed, incomplete mental graphs
    of the world.
    
    Our real intelligence isn't having a perfect model; it's the
    process of learning and adapting that model in the face
    of new evidence. This ability to adapt our causal graph is far more
    powerful than simply having a static, correct one.
    
    This is what a causal framework offers AI. It provides a
    scaffolding for true adaptation.
    """
)

st.markdown("##### A New Example: Learning a Light Switch")
st.markdown(
    """
    1.  **Child's First Model:** A child learns a simple, correct causal model: `Switch_Down -> Light_On`. This model has perfect predictive power.
    
    2.  **Falsifying Evidence:** The child encounters a "three-way switch" (two switches, one light). They press their switch down, but the light *stays off*.
    
    3.  **Statistical vs. Causal AI:**
        -   A **statistical model** trained on "switch position -> light state" would just see its accuracy drop. It would update its weights to predict "Light_On" with 50% probability. It reports "high loss" but doesn't understand *why*.
        -   The **causal child** realizes their *model is wrong*. Their SCM is falsified. They immediately look for *other potential causes* and discover the second switch. Their mental graph adapts from `Switch_A -> Light` to a new, more complex graph: `(Switch_A, Switch_B) -> Light`.
    
    This is the "big leap." A standard AI, when it fails, can only report "high loss."
    A causal AI could pinpoint *why* it failed—"the data violates
    my assumed mechanism for 'Light'"—and use that to *debug*
    and restructure its own internal graph.
    
    As we saw on the Counterfactuals and Principle of Independent Mechanism pages, when our
    assumptions were violated (by a lab result or by dependent noise),
    we didn't just get a "bad score." We got a specific, actionable
    insight: "This graph is wrong."
    
    Whether the causal structures are "actually true" in some deep
    philosophical sense may not be as
    important as this ability to learn and adapt them.
    """
)


st.subheader("Generalization through Causal Representations")
st.markdown(
    """
    This causal approach is also what enables true generalization
    beyond the training data.
    
    If an AI has only ever seen dogs, a statistical model
    learns the pattern for "dog." When it sees a cat, it may
    fail completely, classifying it as a "dog" with low confidence.
    
    But a causal AI would try to learn the independent mechanisms
    that *produce* the data. This is Causal Representation Learning (CRL),
    a major open area of research. It wouldn't just learn a "dog" blob;
    it would try to learn a *compositional model* of independent factors:
    
    -   Mechanism for fur texture
    -   Mechanism for four-legged movement
    -   Mechanism for sound generation (e.g., barking)
    -   Mechanism for behavior (e.g., chasing a ball)
    
    When this AI sees a cat, it can recognize the known components
    (fur, four legs) but also identify a new, unseen component
    (meowing). It can reason that this is a *new combination*
    of mechanisms, allowing it to identify the cat as
    **"not-a-dog"** instead of just failing.
    
    It understands that a cat isn't just a "weird dog" because the
    causal mechanism for its *sound* is fundamentally different,
    even if the "fur" and "legs" mechanisms are similar.
    
    This project just scratched the surface, but it's clear that
    causality provides a formal language not just for robustness,
    but for the very *process of adaptation and generalization*
    that feels so central to intelligence itself.
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