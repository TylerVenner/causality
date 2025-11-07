# Interactive Causal Inference

An interactive Streamlit application for learning the fundamental concepts of causal inference, from Structural Causal Models (SCMs) to discovery algorithms.

This app is the result of a two-month study of *Elements of Causal Inference* by Jonas Peters, Dominik Janzing, and Bernhard Schölkopf.
It uses interactive simulations to build hands-on intuition for this powerful and exciting field.

---

## 🚀 Quick Start

Run the application locally in three simple steps:

```bash
# 1. Clone the repository
git clone https://github.com/TylerVenner/causality.git
cd causality

# 2. Install the required dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run Welcome.py
```

---

## 🗺️ The Learning Journey

The app is structured as a sequence of interactive pages, each building on the last:

| Step                              | Topic                                           | Description                                                                    |
| --------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------ |
| **0. Introduction**               | Statistical association vs. causal intervention | Explore the difference between ( P(Y \mid X) ) and ( P(Y \mid \text{do}(X)) ). |
| **1. Asymmetry of Interventions** | Seeing vs. doing                                | Discover how interventions break statistical symmetry.                         |
| **2. Simulating Counterfactuals** | "What if?" questions                            | Model specific hypothetical changes to past events.                            |
| **3. Independence of Mechanism**  | Discovering cause from effect                   | Learn why causal mechanisms are stable and independent.                        |
| **4. Confounding vs. Mediation**  | When to control for a variable                  | Understand back-door and front-door criteria.                                  |
| **5. The Causal Markov Property** | Graph structure and independence                | How d-separation encodes conditional independencies.                           |
| **6. PC Algorithm**               | Constraint-based discovery                      | Discover causal graphs under causal sufficiency.                               |
| **7. Hidden Confounding & FCI**   | When assumptions fail                           | Handle unobserved variables using the FCI algorithm.                           |
| **8. Conclusion**                 | Reflection                                      | Summarize key insights from the journey.                                       |

---

## 💡 Key Concepts Explored

* **Structural Causal Models (SCMs)**
* **The do()-operator** and interventions
* **Counterfactuals** (Abduction, Action, Prediction)
* **Principle of Independent Mechanisms (PIM)**
* **Additive Noise Models (ANMs)**
* **Confounding** (Back-Door Paths) vs. **Mediation**
* **Causal Markov Property** & **d-Separation**
* **Faithfulness Assumption**
* **Causal Sufficiency**
* **PC Algorithm**
* **FCI Algorithm** & **Partial Ancestral Graphs (PAGs)**

---

## 📄 License

This project is licensed under the **MIT License**.