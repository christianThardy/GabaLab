# GabaLab 🪫

GabaLab is a toolkit built on top of <a href="https://github.com/TransformerLensOrg/TransformerLens" title="TransformerLens" rel="nofollow">transformer-lens</a> for weights-based and activation-based interpretability of internal suppression circuits in transformer models. It's primary focuses are:

- Eigenvalue spectrum analysis to investigate the eigenvalue distribution of OV circuits to get insight into underlying attention mechanism dynamics.

- Token-level sensitivity & dimensionality analysis to explore token representations and perform Jacobian-based sensitivity studies to understand the impact of individual tokens.

- Cross-model comparative analysis to compare attention patterns and circuit functionality across different transformer models.

# Kickstart

Has all the same functionality as a HookedTransformer from transformer-lens. Quick example of how to run a cross-model eigenvalue comparison:

```python
from gaba import CircuitAnalyzer

import plotly.io as pio
pio.renderers.default = "notebook"

# Initialize the analyzer with your model, prompt, and answer tokens
analyzer = CircuitAnalyzer(model, prompt, answer_tokens)

# Define model-layer-head specifications
# Key values are a list of tuples which represent:
# (layer, head, description)
model_specs = {
    "EleutherAI/gpt-neo-1.3B": [
        (3, 3, "Early Layer, Early Head"),
        (23, 13, "Late Layer, Late Head")
    ],
    "EleutherAI/pythia-1.4b-v0": [
        (4, 4, "Early Layer, Early Head"),
        (16, 9, "Late Layer, Late Head")
    ],
    "google/gemma-2-2b": [
        (12, 6, "Early Layer, Early Head"),
        (16, 2, "Late Layer, Early Head")
    ],
    "meta-llama/Llama-3.2-3B": [ 
        (4, 9, "Early Layer, Late Head"),
        (26, 22, "Late Layer, Late Head")
    ],
    "mistral-7b": [
        (29, 9, "Late Layer, Late Head"),
        (30, 2, "Late Layer, Early Head")
    ],
    "meta-llama/Meta-Llama-3-8B": [
        (31, 14, "ToM Suppressor"),
        (30, 31, "Terminal Head")
    ],
    "google/gemma-2-9b-it": [
        (16, 1, "ToM Suppressor"),
        (23, 5, "Terminal Head")
    ],
    "Qwen/Qwen2.5-14B-Instruct": [
        (16, 1, "ToM Suppressor"),
        (23, 5, "Terminal Head")
    ],
}

# Run the cross-model eigenvalue comparison
results_df, figures = analyzer.run_cross_model_eigenvalue_comparison(model_specs)

# Display and save the generated figures
for name, fig in figures.items():
    filename = f"{name.replace('/', '_')}.html"
    fig.write_html(filename)
    print(f"Figure saved to: {filename}")
```
