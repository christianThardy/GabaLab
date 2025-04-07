import os
import json
import gc
import random
import traceback
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from scipy import linalg
from scipy.sparse.linalg import eigs

from transformer_lens import HookedTransformer

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm


class CircuitAnalyzer:
    """    
    Provides a unified interface for investigating attention heads and their
    functional roles in transformer models through various analysis methods:
    
    1. Component-level analysis (Q, K, V, O circuits)
    2. Head ablation and intervention studies
    3. Head-type detection and analysis
    4. Eigenvalue spectrum analysis of OV circuits
    5. Token representation and dimensionality analysis
    6. Cross-model comparative analysis
    7. Token-level Jacobian sensitivity
    
    The analyzer supports multiple decoder-only model architectures and enables both activation-based
    and weights-based analysis approaches.
    """
    def __init__(self, model, dataset=None, answer_tokens=None):
        """
        Initialize the analyzer with a model and dataset.

        Args:
            model: The transformer model to analyze
            dataset: The dataset containing prompts for analysis (optional)
            answer_tokens: Token IDs for the answers in the dataset (optional)
        """
        self.model = model
        self.dataset = dataset
        self.answer_tokens = answer_tokens

        # Get device
        try:
            self.device = next(model.parameters()).device
        except:
            self.device = torch.device('cpu')

        # Check if model is supported
        if not hasattr(model, 'cfg'):
            raise ValueError("Model must have a 'cfg' attribute (HookedTransformer)")

        # Print model architecture info
        print(f"Model: {getattr(model, 'name', type(model).__name__)}")
        print(f"Layers: {model.cfg.n_layers}")
        print(f"Heads: {model.cfg.n_heads}")
        print(f"Hidden size: {model.cfg.d_model}")
        print(f"Head dimension: {model.cfg.d_head}")

    # Sequence-level log probability functions ---
    def sequence_log_prob(self, prompt, answer_tokens):
        """
        Computes the total log probability of generating the answer_tokens given the prompt.

        Args:
            prompt: Tensor of shape [1, prompt_seq] (assumed single example)
            answer_tokens: A list or tensor of token IDs for the answer (e.g. multi-token answer)

        Returns:
            A scalar tensor containing the sum of log probabilities for the answer tokens.
        """
        if not torch.is_tensor(answer_tokens):
            answer_tokens = torch.tensor(answer_tokens, device=prompt.device)
        if answer_tokens.dim() == 1:
            answer_tokens = answer_tokens.unsqueeze(0)
        # Concatenate prompt and answer tokens
        full_seq = torch.cat([prompt, answer_tokens], dim=1)  # [1, prompt_len + answer_len]
        with torch.no_grad():
            logits = self.model(full_seq)  # [1, seq, vocab]
        prompt_len = prompt.shape[1]
        log_probs = []
        # For each answer token, use logits at the position before its token
        for i in range(answer_tokens.shape[1]):
            pos = prompt_len + i - 1  # prediction position for answer token i
            logits_at_pos = logits[0, pos, :]
            token_id = answer_tokens[0, i]
            token_log_prob = logits_at_pos[token_id] - torch.logsumexp(logits_at_pos, dim=0)
            log_probs.append(token_log_prob)
        return torch.stack(log_probs).sum()

    def sequence_logit_diff(self, prompt, correct_answer_tokens, incorrect_answer_tokens):
        """
        Computes the difference in total log probability between the correct and incorrect answers,
        given a prompt.

        Args:
            prompt: Tensor of shape [1, prompt_seq].
            correct_answer_tokens: List/tensor of token IDs for the correct answer.
            incorrect_answer_tokens: List/tensor of token IDs for the incorrect answer.

        Returns:
            A scalar tensor: (log probability for correct answer) - (log probability for incorrect answer)
        """
        correct_log_prob = self.sequence_log_prob(prompt, correct_answer_tokens)
        incorrect_log_prob = self.sequence_log_prob(prompt, incorrect_answer_tokens)
        return correct_log_prob - incorrect_log_prob

    def get_valid_layer_head(self, layer, head):
        """
        Adjust layer and head indices to be valid for the current model.

        Args:
            layer: Layer index (possibly out of range)
            head: Head index (possibly out of range)

        Returns:
            Tuple of (valid_layer, valid_head)
        """
        max_layer = self.model.cfg.n_layers - 1
        max_head = self.model.cfg.n_heads - 1

        valid_layer = min(layer, max_layer)
        valid_head = min(head, max_head)

        if valid_layer != layer or valid_head != head:
            print(f"Warning: Adjusted layer from {layer} to {valid_layer} and head from {head} to {valid_head} to fit model architecture.")

        return valid_layer, valid_head

    def get_hook_name(self, layer, component):
        """
        Get the correct hook name for a given layer and component.

        Args:
            layer: Layer index
            component: Component name (q, k, v, z, pattern, attn_out)

        Returns:
            Correct hook name for the model
        """
        # Check if the model uses 'blocks' (Gemma, LLAMA) or other format
        if hasattr(self.model, 'blocks'):
            prefix = f"blocks.{layer}"
        else:
            # Model from Huggingface typically use this format
            prefix = f"blocks.{layer}" if hasattr(self.model, 'blocks') else f"transformer.h.{layer}"

        # Handle different component names
        component_mapping = {
            'q': f"{prefix}.attn.hook_q",
            'k': f"{prefix}.attn.hook_k",
            'v': f"{prefix}.attn.hook_v",
            'z': f"{prefix}.attn.hook_z",
            'pattern': f"{prefix}.attn.hook_pattern",
            'attn_out': f"{prefix}.hook_attn_out"
        }

        # Return the hook name if it exists
        if component in component_mapping:
            return component_mapping[component]
        else:
            raise ValueError(f"Unknown component: {component}")

    def extract_head_components(self, tokens, layer, head, return_cache=False):
        """
        Extract Q, K, V, and O components for a specific head in a model-agnostic way.

        Args:
            tokens: Input tokens to the model
            layer: Layer index
            head: Head index
            return_cache: Whether to return the full cache

        Returns:
            Dict containing Q, K, V, O, attention patterns, and optionally the full cache
        """
        # Adjust layer and head to be valid for the model
        layer, head = self.get_valid_layer_head(layer, head)

        # Determine components to extract
        components_to_extract = ['q', 'k', 'v', 'z', 'pattern', 'attn_out']
        hook_names = []

        # Build a list of hook names to extract
        for component in components_to_extract:
            try:
                hook_name = self.get_hook_name(layer, component)
                hook_names.append(hook_name)
            except Exception as e:
                print(f"Warning: Could not get hook name for {component}: {e}")

        # Define a filter function for run_with_cache
        def names_filter(name):
            return any(hook_name in name for hook_name in hook_names)

        # Run model to get activations
        try:
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=names_filter
            )
        except Exception as e:
            print(f"Error running model with cache: {e}")
            if return_cache:
                return {"cache": None}
            return {}

        # Extract components
        components = {}

        # Process each component from the cache
        for component in components_to_extract:
            try:
                hook_name = self.get_hook_name(layer, component)

                if hook_name in cache:
                    # Extract based on component type
                    if component == 'q':
                        q = cache[hook_name]
                        components["q"] = q[:, :, head] if q.ndim == 4 else None

                    elif component == 'k':
                        k = cache[hook_name]
                        # Handle different models' KV sharing schemes
                        n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', k.shape[2])
                        kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                        components["k"] = k[:, :, kv_head] if k.ndim == 4 else None

                    elif component == 'v':
                        v = cache[hook_name]
                        # Handle different models' KV sharing schemes
                        n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', v.shape[2])
                        kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                        components["v"] = v[:, :, kv_head] if v.ndim == 4 else None

                    elif component == 'z':
                        z = cache[hook_name]
                        components["z"] = z[:, :, head] if z.ndim == 4 else None

                    elif component == 'pattern':
                        pattern = cache[hook_name]
                        # Handle different pattern shapes
                        if pattern.ndim == 3:  # [batch, seq_q, seq_k]
                            components["pattern"] = pattern
                        elif pattern.ndim == 4:  # [batch, head, seq_q, seq_k]
                            components["pattern"] = pattern[:, head]
                        else:
                            components["pattern"] = pattern[:, :, head]

                    elif component == 'attn_out':
                        components["attn_out"] = cache[hook_name]

            except Exception as e:
                print(f"Warning: Could not extract {component}: {e}")

        # Try to extract weight matrices (model-specific)
        try:
            # Get weights through architecture-specific paths
            if hasattr(self.model, 'blocks'):
                # Gemma, LLAMA style
                W_O = self.model.blocks[layer].attn.W_O[head].detach()
                W_Q = self.model.blocks[layer].attn.W_Q[head].detach()

                # Handle different KV weight schemes
                if hasattr(self.model.blocks[layer].attn, 'W_K'):
                    n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', self.model.cfg.n_heads)
                    kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                    W_K = self.model.blocks[layer].attn.W_K[kv_head].detach()
                    W_V = self.model.blocks[layer].attn.W_V[kv_head].detach()
                elif hasattr(self.model.blocks[layer].attn, 'W_KV'):
                    n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', self.model.cfg.n_heads)
                    kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                    W_KV = self.model.blocks[layer].attn.W_KV[kv_head].detach()
                    d_head = self.model.cfg.d_head
                    W_K = W_KV[:, :d_head]
                    W_V = W_KV[:, d_head:]
                else:
                    W_K = None
                    W_V = None

            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # GPT-2 style
                # These are approximate - may need adjusting for certain models
                W_O = self.model.transformer.h[layer].attn.c_proj.weight.view(
                    self.model.cfg.n_heads, self.model.cfg.d_head, self.model.cfg.d_model
                )[head].detach()

                # This is an approximation and might need adjustment
                W_Q = self.model.transformer.h[layer].attn.c_attn.weight[:, :self.model.cfg.d_model].view(
                    self.model.cfg.d_model, self.model.cfg.n_heads, self.model.cfg.d_head
                )[:, head, :].detach()

                # Simplified - just capture that we attempted to get weights
                W_K = None
                W_V = None

            else:
                # Unknown architecture - set to None
                W_O = None
                W_Q = None
                W_K = None
                W_V = None

            # Store weight matrices
            components["W_O"] = W_O
            components["W_Q"] = W_Q
            components["W_K"] = W_K
            components["W_V"] = W_V

            # Calculate head output contribution if possible
            if "z" in components and "W_O" in components and components["z"] is not None and components["W_O"] is not None:
                head_output = torch.matmul(components["z"], components["W_O"])
                components["output"] = head_output

        except Exception as e:
            print(f"Warning: Could not extract weight matrices: {e}")
            components["W_O"] = None
            components["W_Q"] = None
            components["W_K"] = None
            components["W_V"] = None

        if return_cache:
            components["cache"] = cache

        return components

        # Add this helper function to the analyzer class
    def get_dynamic_hook_name(self, layer, component):
        # Try a blocks-based naming scheme
        prefix = f"blocks.{layer}."
        candidate_keys = [key for key in self.model.cfg.hook_names if key.startswith(prefix) and component in key]
        # If none found, try transformer.h-based naming scheme
        if not candidate_keys:
            prefix = f"transformer.h.{layer}."
            candidate_keys = [key for key in self.model.cfg.hook_names if key.startswith(prefix) and component in key]
        if not candidate_keys:
            raise ValueError(f"No hook key found for component {component} at layer {layer}")
        return candidate_keys[0]

    # Then replace the ablate_head_components method with this version:
    def ablate_head_components(self, tokens, layer, head, ablate_q=False, ablate_k=False,
                           ablate_v=False, ablate_o=False, return_logits=True):
        hooks = []
        # For each component, try to get a dynamic hook name; if that fails, fall back.
        if ablate_q:
            # Capture the current head value with a default argument (h=head)
            def hook_q(q, hook, h=head):
                q[:, :, h] = q[:, :, h].mean(dim=(0, 1), keepdim=True)
                return q
            try:
                hook_name = self.get_dynamic_hook_name(layer, 'hook_q')
            except Exception as e:
                hook_name = f"blocks.{layer}.attn.hook_q"
            hooks.append((hook_name, hook_q))
        if ablate_k:
            def hook_k(k, hook, h=head):
                n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', k.shape[2])
                kv_head = min(h * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                k[:, :, kv_head] = k[:, :, kv_head].mean(dim=(0, 1), keepdim=True)
                return k
            try:
                hook_name = self.get_dynamic_hook_name(layer, 'hook_k')
            except Exception as e:
                hook_name = f"blocks.{layer}.attn.hook_k"
            hooks.append((hook_name, hook_k))
        if ablate_v:
            def hook_v(v, hook, h=head):
                n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', v.shape[2])
                kv_head = min(h * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                v[:, :, kv_head] = v[:, :, kv_head].mean(dim=(0, 1), keepdim=True)
                return v
            try:
                hook_name = self.get_dynamic_hook_name(layer, 'hook_v')
            except Exception as e:
                hook_name = f"blocks.{layer}.attn.hook_v"
            hooks.append((hook_name, hook_v))
        if ablate_o:
            def hook_z(z, hook, h=head):
                z[:, :, h] = z[:, :, h].mean(dim=(0, 1), keepdim=True)
                return z
            try:
                hook_name = self.get_dynamic_hook_name(layer, 'hook_z')
            except Exception as e:
                hook_name = f"blocks.{layer}.attn.hook_z"
            hooks.append((hook_name, hook_z))
        self.model.reset_hooks()
        for hook_name, hook_fn in hooks:
            self.model.add_hook(hook_name, hook_fn)
        with torch.no_grad():
            outputs = self.model(tokens) if return_logits else self.model(tokens, return_type=None)
        self.model.reset_hooks()
        return outputs

    def logits_to_ave_logit_diff(self, logits, answer_tokens):
        """
        Calculate average logit difference between correct and incorrect answers.

        Args:
            logits: Model output logits [batch, seq, vocab]
            answer_tokens: Token IDs for answers [batch, 2] or [2]

        Returns:
            Average logit difference
        """
        # Ensure we have logits
        if logits is None:
            print("Warning: Logits is None, returning 0")
            return torch.tensor(0.0, device=self.device)

        # Get the batch size from logits
        logits_batch_size = logits.shape[0]

        # Check if answer_tokens are provided
        if answer_tokens is None:
            print("Warning: answer_tokens is None, returning 0")
            return torch.tensor(0.0, device=self.device)

        # Ensure answer_tokens are on the same device as logits
        if isinstance(answer_tokens, torch.Tensor):
            answer_tokens = answer_tokens.to(logits.device)

            if answer_tokens.dim() == 1:
                # If answer_tokens is [2], expand to [batch, 2]
                answer_tokens = answer_tokens.unsqueeze(0).expand(logits_batch_size, -1)
            elif answer_tokens.shape[0] != logits_batch_size:
                # If batch sizes don't match, take only what we need or repeat
                if answer_tokens.shape[0] > logits_batch_size:
                    answer_tokens = answer_tokens[:logits_batch_size]
                else:
                    # Repeat the tokens to match the batch size
                    repeats = (logits_batch_size + answer_tokens.shape[0] - 1) // answer_tokens.shape[0]
                    answer_tokens = answer_tokens.repeat(repeats, 1)[:logits_batch_size]
        try:
            # Extract final logits
            final_logits = logits[:, -1, :]

            # Gather logits for the answer tokens
            answer_logits = final_logits.gather(dim=-1, index=answer_tokens)

            # Calculate logit difference
            correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
            answer_logit_diff = correct_logits - incorrect_logits

            return answer_logit_diff.mean()
        except Exception as e:
            print(f"Error calculating logit difference: {e}")
            return torch.tensor(0.0, device=self.device)

    # Use with new component analysis code
    def evaluate_ablation_effect(self, tokens, layer, head, ablation_types, answer_tokens=None):
        """
        Evaluate the effect of different types of ablations on model performance
        using sequence-level log probability difference.
        """
        if answer_tokens is None:
            answer_tokens = self.answer_tokens
        if answer_tokens is None:
            raise ValueError("answer_tokens must be provided")

        # Compute baseline logit difference using sequence_logit_diff.
        with torch.no_grad():
            baseline_diff = self.sequence_logit_diff(tokens, answer_tokens[:, 0], answer_tokens[:, 1])
        results = []
        for ablation_type in ablation_types:
            ablate_q = "q" in ablation_type
            ablate_k = "k" in ablation_type
            ablate_v = "v" in ablation_type
            ablate_o = "o" in ablation_type or "all" in ablation_type
            try:
                ablated_logits = self.ablate_head_components(tokens, layer, head,
                                                              ablate_q=ablate_q, ablate_k=ablate_k,
                                                              ablate_v=ablate_v, ablate_o=ablate_o)
                ablated_diff = self.sequence_logit_diff(tokens, answer_tokens[:, 0], answer_tokens[:, 1])
                effect = baseline_diff - ablated_diff
                results.append({
                    "Layer": layer,
                    "Head": head,
                    "Ablation Type": ablation_type,
                    "Baseline Logit Diff": baseline_diff.item(),
                    "Ablated Logit Diff": ablated_diff.item(),
                    "Effect": effect.item(),
                    "Normalized Effect": effect.item() / baseline_diff.item() if baseline_diff.item() != 0 else 0
                })
            except Exception as e:
                print(f"Error evaluating ablation effect for {ablation_type}: {e}")
                results.append({
                    "Layer": layer,
                    "Head": head,
                    "Ablation Type": ablation_type,
                    "Baseline Logit Diff": baseline_diff.item(),
                    "Ablated Logit Diff": 0,
                    "Effect": 0,
                    "Normalized Effect": 0
                })
        return pd.DataFrame(results)

    def run_fine_grained_ov_qk_ablations(self, tokens, layer, head, answer_tokens=None):
        """
        Run detailed OV-QK ablation experiments for a specific head.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            answer_tokens: Token IDs for answers

        Returns:
            DataFrame with detailed ablation results
        """
        if answer_tokens is None:
            answer_tokens = self.answer_tokens

        if answer_tokens is None:
            raise ValueError("answer_tokens must be provided")

        # Define a comprehensive set of ablation types
        ablation_types = [
            "q", "k", "v", "o",            # Individual components
            "qk", "qv", "ko", "vo",        # Pairs
            "qkv", "qko", "qvo", "kvo",    # Triples
            "qkvo"                         # All components
        ]

        return self.evaluate_ablation_effect(tokens, layer, head, ablation_types, answer_tokens)

    def plot_fine_grained_ablation_results(self, ablation_df):
        """
        Plot fine-grained ablation results.

        Args:
            ablation_df: DataFrame from run_fine_grained_ov_qk_ablations
        """
        # Check if plotly is available        
        if ablation_df.empty:
            print("No data to visualize.")
            return

        # Sort by effect size
        ablation_df = ablation_df.sort_values("Effect", ascending=False)

        # Create basic bar chart
        fig = px.bar(
            ablation_df,
            x="Ablation Type",
            y="Effect",
            title=f"Ablation Effects for L{ablation_df['Layer'].iloc[0]}H{ablation_df['Head'].iloc[0]}",
            labels={"Effect": "Change in Logit Diff", "Ablation Type": "Ablated Components"},
            color="Effect",
            color_continuous_scale="RdBu_r"
        )

        fig.update_layout(
            xaxis_title="Ablated Components",
            yaxis_title="Change in Logit Difference",
            width=800,
            height=500
        )

        fig.show()

        # Create OV vs QK comparison
        fig = go.Figure()

        # Group ablation types
        qk_types = [t for t in ablation_df["Ablation Type"] if ("q" in t or "k" in t) and "o" not in t and "v" not in t]
        ov_types = [t for t in ablation_df["Ablation Type"] if ("o" in t or "v" in t) and "q" not in t and "k" not in t]
        mixed_types = [t for t in ablation_df["Ablation Type"] if t not in qk_types and t not in ov_types]

        # Filter dataframe
        qk_df = ablation_df[ablation_df["Ablation Type"].isin(qk_types)]
        ov_df = ablation_df[ablation_df["Ablation Type"].isin(ov_types)]
        mixed_df = ablation_df[ablation_df["Ablation Type"].isin(mixed_types)]

        # Add traces if data exists
        if not qk_df.empty:
            fig.add_trace(go.Bar(
                x=qk_df["Ablation Type"],
                y=qk_df["Effect"],
                name="QK Components",
                marker_color="blue"
            ))

        if not ov_df.empty:
            fig.add_trace(go.Bar(
                x=ov_df["Ablation Type"],
                y=ov_df["Effect"],
                name="OV Components",
                marker_color="red"
            ))

        if not mixed_df.empty:
            fig.add_trace(go.Bar(
                x=mixed_df["Ablation Type"],
                y=mixed_df["Effect"],
                name="Mixed Components",
                marker_color="purple"
            ))

        fig.update_layout(
            title=f"OV vs QK Ablation Effects for L{ablation_df['Layer'].iloc[0]}H{ablation_df['Head'].iloc[0]}",
            xaxis_title="Ablated Components",
            yaxis_title="Change in Logit Difference",
            barmode="group",
            width=900,
            height=500
        )

        fig.show()

    def compute_qk_similarity(self, q, k):
        """
        Compute the similarity between Q and K vectors.

        Args:
            q: Query vectors [batch, seq, d_head]
            k: Key vectors [batch, seq, d_head]

        Returns:
            Similarity matrix [batch, seq, seq]
        """        
        # Normalize q and k for cosine similarity
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # Compute similarity
        return torch.bmm(q_norm, k_norm.transpose(1, 2))

    def compute_ov_effect(self, v, W_O):
        """
        Compute the effect of OV transformation.

        Args:
            v: Value vectors [batch, seq, d_head]
            W_O: Output weight matrix [d_head, d_model]

        Returns:
            OV effect [batch, seq, d_model]
        """
        return torch.matmul(v, W_O)

    def visualize_qk_ov_comparison(self, layer, head, tokens, token_strs=None):
        """
        Compare QK and OV patterns for a specific head.

        Args:
            layer: Layer index
            head: Head index
            tokens: Input tokens
            token_strs: Token strings for labels
        """
        # Extract components
        components = self.extract_head_components(tokens, layer, head)

        # Check if this is what is needed
        if "q" not in components or "k" not in components or "v" not in components or "W_O" not in components:
            print("Missing required components for visualization.")
            return

        if components["q"] is None or components["k"] is None or components["v"] is None or components["W_O"] is None:
            print("One or more required components are None.")
            return

        # Generate token strings if not provided
        if token_strs is None:
            try:
                token_strs = [self.model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]
            except Exception as e:
                print(f"Could not generate token strings: {e}")
                token_strs = [f"Token {i}" for i in range(tokens.shape[1])]

        # Compute QK similarity
        try:
            # Use the helper function
            qk_sim = self.compute_qk_similarity(components["q"], components["k"])
        except Exception as e:
            print(f"Error computing QK similarity: {e}")
            return

        # Compute OV effect visualization
        try:
            # Use the helper function
            ov_effect = self.compute_ov_effect(components["v"], components["W_O"])

            # Reshape for PCA [seq, d_model]
            ov_flat = ov_effect[0].reshape(ov_effect.shape[1], -1)

            # Apply PCA
            pca = PCA(n_components=2)
            ov_pca = pca.fit_transform(ov_flat.cpu().numpy())
        except Exception as e:
            print(f"Error computing OV effect: {e}")
            return

        # Create the plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"QK Similarity (L{layer}H{head})", f"OV Effect PCA (L{layer}H{head})"]
        )

        # QK heatmap
        fig.add_trace(
            go.Heatmap(
                z=qk_sim[0].cpu().numpy(),
                x=token_strs,
                y=token_strs,
                colorscale="Viridis"
            ),
            row=1, col=1
        )

        # OV effect scatter
        fig.add_trace(
            go.Scatter(
                x=ov_pca[:, 0],
                y=ov_pca[:, 1],
                mode="markers+text",
                text=token_strs,
                textposition="top center",
                marker=dict(size=10, color=list(range(len(ov_pca))), colorscale="Viridis")
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=600,
            width=1200,
            title_text=f"QK and OV Analysis for Layer {layer}, Head {head}"
        )

        fig.show()

        return components

    def classify_head(self, tokens, layer, head, token_strs=None):
        """
        Classify a head by analyzing its attention patterns and behavior.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            token_strs: Token strings

        Returns:
            Classification results and scores for the head
        """
        # Generate token strings if not provided
        if token_strs is None:
            try:
                token_strs = [self.model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]
            except Exception as e:
                print(f"Could not generate token strings: {e}")
                token_strs = [f"Token {i}" for i in range(tokens.shape[1])]

        # Extract components
        components = self.extract_head_components(tokens, layer, head)

        # Check if we have the pattern component
        if "pattern" not in components or components["pattern"] is None:
            print("Warning: No attention pattern available for classification.")
            return {
                "layer": layer,
                "head": head,
                "classification": "Unknown",
                "scores": {
                    "Previous Token Head": 0,
                    "Duplicate Token Head": 0,
                    "Induction Head": 0,
                    "Copy Suppression Head": 0
                }
            }

        pattern = components["pattern"][0].cpu()  # Take first batch

        # Analyze previous token behavior
        prev_token_score = 0
        try:
            # For a previous token head, we expect high attention to positions i-1
            diagonal_prev = torch.diagonal(pattern, offset=-1)
            avg_prev_attention = diagonal_prev.mean().item()
            avg_attention = pattern.mean().item()
            prev_token_score = avg_prev_attention / avg_attention if avg_attention != 0 else 0
        except Exception as e:
            print(f"Warning: Could not analyze previous token behavior: {e}")

        # Analyze duplicate token behavior
        duplicate_token_score = 0
        try:
            # Build a map of duplicate tokens
            token_positions = {}
            duplicate_positions = []

            for i, token in enumerate(token_strs):
                if token in token_positions:
                    duplicate_positions.append((i, token_positions[token]))
                    token_positions[token].append(i)
                else:
                    token_positions[token] = [i]

            # Calculate attention between duplicate tokens
            duplicate_attention_scores = []

            for curr_pos, prev_positions in duplicate_positions:
                for prev_pos in prev_positions:
                    if prev_pos < curr_pos:  # Only consider previous occurrences
                        attention_score = pattern[curr_pos, prev_pos].item()
                        duplicate_attention_scores.append(attention_score)

            avg_duplicate_attention = np.mean(duplicate_attention_scores) if duplicate_attention_scores else 0
            duplicate_token_score = avg_duplicate_attention / pattern.mean().item() if pattern.mean().item() != 0 else 0
        except Exception as e:
            print(f"Warning: Could not analyze duplicate token behavior: {e}")

        # Analyze induction head behavior
        induction_score = 0
        try:
            # For induction heads, check for attention to tokens that follow occurrences of the same token
            token_positions = {}
            induction_patterns = []

            for i, token in enumerate(token_strs):
                if token in token_positions:
                    for prev_pos in token_positions[token]:
                        if prev_pos + 1 < len(token_strs) and i > prev_pos + 1:
                            # Check attention from current token to token after previous occurrence
                            induction_patterns.append((i, prev_pos + 1, pattern[i, prev_pos + 1].item()))

                # Update positions
                if token in token_positions:
                    token_positions[token].append(i)
                else:
                    token_positions[token] = [i]

            avg_induction_attention = np.mean([x[2] for x in induction_patterns]) if induction_patterns else 0
            induction_score = avg_induction_attention / pattern.mean().item() if pattern.mean().item() != 0 else 0
        except Exception as e:
            print(f"Warning: Could not analyze induction head behavior: {e}")

        # Analyze copy suppression behavior
        copy_suppression_score = 0
        try:
            # Check for copy suppression behavior
            if "W_O" in components and components["W_O"] is not None and self.answer_tokens is not None:
                ablation_results = self.evaluate_ablation_effect(
                    tokens, layer, head, ["o"], self.answer_tokens
                )

                # For copy suppression, ablating should increase logit diff
                if not ablation_results.empty:
                    effect = ablation_results["Effect"].iloc[0]
                    copy_suppression_score = -effect if effect < 0 else 0
            else:
                # Fall back to a simple estimate
                if "v" in components and "W_O" in components and components["v"] is not None and components["W_O"] is not None:
                    ov_output = torch.matmul(components["v"], components["W_O"])
                    final_token_output = ov_output[0, -1].cpu()
                    negative_ratio = (final_token_output < 0).float().mean().item()
                    copy_suppression_score = negative_ratio
        except Exception as e:
            print(f"Warning: Could not analyze copy suppression behavior: {e}")

        # Compile scores
        scores = {
            "Previous Token Head": prev_token_score,
            "Duplicate Token Head": duplicate_token_score,
            "Induction Head": induction_score,
            "Copy Suppression Head": copy_suppression_score
        }

        # Determine the most likely classification
        classification = max(scores, key=scores.get) if any(scores.values()) else "Unknown"

        # Return classification results
        return {
            "layer": layer,
            "head": head,
            "classification": classification,
            "scores": scores
        }

    def analyze_copy_suppression_behavior(self, tokens, layer, head):
        """
        Analyze whether a head behaves as a copy suppression head using the methods described in
        McDougall et al. copy suppression paper.
        """
        # 1. Run ablation to measure effect on believed vs. actual location
        ablation_results = self.evaluate_ablation_effect(
            tokens, layer, head, ["o"], self.answer_tokens
        )

        # For copy suppression, ablating the head should increase the logit diff
        effect = ablation_results["Effect"].iloc[0]

        # 2. Extract components for OV analysis
        components = self.extract_head_components(tokens, layer, head)

        # 3. Analyze the OV circuit - check if the head suppresses tokens it attends to
        W_O = components["W_O"]  # [d_head, d_model]
        # Print shapes for debugging
        # print(f"W_O shape: {W_O.shape}")

        # Get effective embedding (using MLP0 output if available, otherwise use embedding directly)
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "mlp"):
            MLP0 = self.model.blocks[0].mlp
            WE = self.model.W_E.detach()
            # print(f"WE shape: {WE.shape}")
            effective_embeddings = []
            batch_size = 100
            for i in range(0, WE.shape[0], batch_size):
                batch = WE[i:i+batch_size]
                with torch.no_grad():
                    effective_batch = MLP0(batch)
                effective_embeddings.append(effective_batch.detach())
            effective_WE = torch.cat(effective_embeddings, dim=0)
        else:
            effective_WE = self.model.W_E.detach()

        # print(f"effective_WE shape: {effective_WE.shape}")

        # Compute projected unembedding matrix for proper dimension matching
        W_U = self.model.W_U.detach()  # [d_model, vocab_size]
        # Compute projected unembedding matrix: [d_head, vocab_size]
        W_U_proj = torch.matmul(W_O, W_U)  # [d_head, vocab_size]
        W_U_proj_T = W_U_proj.T  # [vocab_size, d_head]

        # Sample tokens for efficiency
        vocab_size = min(W_U_proj_T.shape[0], effective_WE.shape[0])
        max_tokens = min(2000, vocab_size)

        # Create random indices
        token_indices = torch.randperm(vocab_size, device=W_U.device)[:max_tokens]

        # Sample embeddings
        sampled_WE = effective_WE[token_indices]

        # Project embeddings through W_O: [tokens, d_model] @ [d_head, d_model].T = [tokens, d_head]
        projected_embeddings = torch.matmul(sampled_WE, W_O.t())

        # Compute OV circuit using the projected unembedding matrix
        sampled_W_U = W_U_proj_T[token_indices]
        try:
            ov_circuit = torch.matmul(projected_embeddings, sampled_W_U.t())
            # Extract diagonal elements
            diag_values = torch.diag(ov_circuit)
            diag_negative_ratio = (diag_values < 0).float().mean().item()
            print(f"Diagonal negative ratio: {diag_negative_ratio:.4f}")

            bottom_5_percent_count = 0
            top_10_negative_count = 0

            for i in range(len(token_indices)):
                diag_val = ov_circuit[i, i]
                col = ov_circuit[:, i]
                threshold_idx = max(1, int(0.05 * len(col)))
                if torch.topk(col, threshold_idx, largest=False)[0][-1] >= diag_val:
                    bottom_5_percent_count += 1
                if torch.topk(col, min(10, len(col)), largest=False)[0][-1] >= diag_val:
                    top_10_negative_count += 1

            bottom_5_percent_ratio = bottom_5_percent_count / len(token_indices)
            top_10_negative_ratio = top_10_negative_count / len(token_indices)
            print(f"Bottom 5% ratio: {bottom_5_percent_ratio:.4f}")
        except Exception as e:
            bottom_5_percent_ratio = 0
            top_10_negative_ratio = 0
            diag_negative_ratio = 0
            print(f"Circuit computation failed: {e}")            
            traceback.print_exc()

        # 5. Analyze OV transformation on the last token output
        ov_output = self.compute_ov_effect(components["v"], components["W_O"])
        final_token_output = ov_output[0, -1].cpu()
        negative_ratio = (final_token_output < 0).float().mean().item()
        print(f"Negative output ratio: {negative_ratio:.4f}")

        # Combine results to determine if this is a copy suppression head
        copy_suppression_score = (
            (0.3 * bottom_5_percent_ratio) +
            (0.2 * diag_negative_ratio) +
            (0.3 * 0.0) +  # diag_max_ratio not computed if QK fails; set to 0
            (0.2 * negative_ratio)
        )

        print(f"Copy suppression score: {copy_suppression_score:.4f}")

        return {
            "ablation_effect": effect,
            "negative_output_ratio": negative_ratio,
            "ov_diag_negative_ratio": diag_negative_ratio,
            "ov_diag_top10_negative_ratio": top_10_negative_ratio,
            "ov_diag_bottom5_percent_ratio": bottom_5_percent_ratio,
            "qk_diag_max_ratio": 0,  # QK analysis skipped if failed
            "copy_suppression_score": copy_suppression_score,
            "is_copy_suppression_head": copy_suppression_score > 0.5
        }

    def visualize_ov_suppression_1(self, tokens, layer, head, token_strs=None, top_k=50, prompt_only=False, include_bos=True):
      """
      Visualize the OV suppression pattern for a head, focusing on diagonal elements
      of the OV circuit matrix.

      Args:
          tokens: Input tokens.
          layer: Layer index.
          head: Head index.
          token_strs: Optional token strings; if None, they are generated from the vocabulary.
          top_k: Number of top tokens to show (default is 50).
          prompt_only: If True, only visualize tokens from the prompt.
          include_bos: If False, filter out the <bos> token from visualization.
      """
      # Adjust layer and head to be valid for the model
      layer, head = self.get_valid_layer_head(layer, head)

      # Extract components
      components = self.extract_head_components(tokens, layer, head)

      if "W_O" not in components or components["W_O"] is None:
          print("W_O component not available.")
          return None

      W_O = components["W_O"]

      try:
          # Get unembedding matrix
          W_U = self.model.W_U.detach()  # [d_model, vocab_size]

          # Compute OV projection
          W_U_proj = torch.matmul(W_O, W_U)  # [d_head, vocab_size]
          W_U_proj_T = W_U_proj.T  # [vocab_size, d_head]

          if prompt_only:
              # Use only tokens from the prompt (first batch)
              prompt_token_ids = tokens[0].tolist()

              # Remove duplicates while preserving order
              seen = set()
              prompt_token_ids = [tid for tid in prompt_token_ids if tid not in seen and not seen.add(tid)]

              # Optionally filter out the <bos> token
              if not include_bos and hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, "bos_token_id"):
                  bos_id = self.model.tokenizer.bos_token_id
                  prompt_token_ids = [tid for tid in prompt_token_ids if tid != bos_id]

              # Build token strings from these prompt token ids
              if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'decode'):
                  token_strs = [self.model.tokenizer.decode([int(tid)]) for tid in prompt_token_ids]
              else:
                  token_strs = [f"Token {tid}" for tid in prompt_token_ids]

              # Use these token ids for visualization
              vis_token_indices = torch.tensor(prompt_token_ids, device=W_U.device)
          else:
              # Build token_strs from the vocabulary subset
              vocab_size = min(W_U_proj_T.shape[0], self.model.W_E.detach().shape[0])
              token_ids = torch.arange(vocab_size, device=W_U.device)

              if token_strs is None:
                  if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'decode'):
                      token_strs = [self.model.tokenizer.decode([int(tid)]) for tid in token_ids.tolist()]
                  else:
                      token_strs = [f"Token {i}" for i in range(vocab_size)]

              # Optionally filter out the <bos> token
              if not include_bos and hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, "bos_token_id"):
                  bos_id = self.model.tokenizer.bos_token_id
                  # Filter both token_ids and token_strs accordingly
                  filtered = [(tid, s) for tid, s in zip(token_ids.tolist(), token_strs) if tid != bos_id]
                  if filtered:
                      token_ids, token_strs = zip(*filtered)
                      token_ids = torch.tensor(token_ids, device=W_U.device)
                      token_strs = list(token_strs)
                  else:
                      token_ids = torch.tensor([], device=W_U.device)
                      token_strs = []

              max_tokens = min(top_k * 5, len(token_ids))
              vis_token_indices = token_ids[:max_tokens]

          # Check if we have valid tokens to visualize
          if vis_token_indices.numel() == 0:
              print("No valid tokens to visualize.")
              return None

          # Get embeddings for these tokens
          WE = self.model.W_E.detach()
          if vis_token_indices.max() >= WE.shape[0]:
              vis_token_indices = vis_token_indices[vis_token_indices < WE.shape[0]]

              # Update token strings if needed
              if len(token_strs) > len(vis_token_indices):
                  token_strs = [token_strs[i] for i in range(len(vis_token_indices))]

          sampled_WE = WE[vis_token_indices]

          # Apply MLP0 if available for effective embeddings
          if hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "mlp"):
              MLP0 = self.model.blocks[0].mlp
              with torch.no_grad():
                  effective_WE = MLP0(sampled_WE).detach()
          else:
              effective_WE = sampled_WE

          # Project embeddings through W_O: [tokens, d_head]
          projected_embeddings = torch.matmul(effective_WE, W_O.t())
          sampled_W_U = W_U_proj_T[vis_token_indices]

          # Compute OV circuit matrix
          ov_matrix = torch.matmul(projected_embeddings, sampled_W_U.t())
          diag_values = torch.diag(ov_matrix)

          # Select top and bottom tokens based on diagonal values
          top_indices = torch.argsort(diag_values, descending=True)[:min(top_k, len(diag_values))]
          bottom_indices = torch.argsort(diag_values)[:min(top_k, len(diag_values))]
          top_tokens = [token_strs[i] for i in top_indices.tolist()]
          bottom_tokens = [token_strs[i] for i in bottom_indices.tolist()]

          # Plot heatmaps for top and bottom tokens
          fig, axes = plt.subplots(1, 2, figsize=(55, 55))

          # Top tokens heatmap
          sns.heatmap(
              ov_matrix[top_indices][:, top_indices].cpu().numpy(),
              annot=True, fmt=".2f", cmap="RdBu_r", center=0,
              ax=axes[0], xticklabels=top_tokens, yticklabels=top_tokens
          )
          axes[0].set_title(f"Top {top_k} Tokens with Highest Diagonal Values (L{layer}H{head})")
          axes[0].set_xlabel("Token")
          axes[0].set_ylabel("Token")

          # Bottom tokens heatmap
          sns.heatmap(
              ov_matrix[bottom_indices][:, bottom_indices].cpu().numpy(),
              annot=True, fmt=".2f", cmap="RdBu_r", center=0,
              ax=axes[1], xticklabels=bottom_tokens, yticklabels=bottom_tokens
          )
          axes[1].set_title(f"Top {top_k} Tokens with Most Negative Diagonal Values (L{layer}H{head})")
          axes[1].set_xlabel("Token")
          axes[1].set_ylabel("Token")

          plt.tight_layout()
          plt.show()

          # Plot histogram of diagonal values
          plt.figure(figsize=(12, 6))
          plt.hist(diag_values.cpu().numpy(), bins=50, alpha=0.7)
          plt.axvline(x=0, color='r', linestyle='--', label='Zero')
          plt.title(f"Distribution of Diagonal Values in OV Circuit (L{layer}H{head})")
          plt.xlabel("Diagonal Value")
          plt.ylabel("Count")
          plt.legend()
          plt.grid(alpha=0.3)
          plt.show()

          # Calculate statistics
          negative_diag_ratio = (diag_values < 0).float().mean().item()
          mean_diagonal = diag_values.mean().item()
          min_diagonal = diag_values.min().item()
          max_diagonal = diag_values.max().item()

          # Print statistics
          print(f"Percentage of negative diagonal values: {negative_diag_ratio*100:.2f}%")
          print(f"Mean diagonal value: {mean_diagonal:.4f}")
          print(f"Min diagonal value: {min_diagonal:.4f}")
          print(f"Max diagonal value: {max_diagonal:.4f}")

          return {
              "negative_diag_ratio": negative_diag_ratio,
              "mean_diagonal": mean_diagonal,
              "min_diagonal": min_diagonal,
              "max_diagonal": max_diagonal,
              "top_tokens": top_tokens,
              "bottom_tokens": bottom_tokens
          }

      except Exception as e:
          print(f"Error in visualize_ov_suppression: {e}")
          traceback.print_exc()

          return {
              "negative_diag_ratio": 0,
              "mean_diagonal": 0,
              "min_diagonal": 0,
              "max_diagonal": 0,
              "top_tokens": [],
              "bottom_tokens": []
          }

    def visualize_ov_suppression_2(self, tokens, layer, head, token_strs=None, top_k=50, prompt_only=False, include_bos=True):
      """
      Visualize the OV suppression pattern for a head, focusing on diagonal elements
      of the OV circuit matrix.

      Args:
          tokens: Input tokens.
          layer: Layer index.
          head: Head index.
          token_strs: Optional token strings; if None, they are generated from the vocabulary.
          top_k: Number of top tokens to show (default increased from 20 to 50).
          prompt_only: If True, only visualize tokens from the prompt.
          include_bos: If False, filter out the <bos> token from visualization.
      """      
      # Compute the projected unembedding matrix for proper dimension matching.
      W_O = self.extract_head_components(tokens, layer, head)["W_O"]
      W_U = self.model.W_U.detach()  # [d_model, vocab_size]
      W_U_proj = torch.matmul(W_O, W_U)  # [d_head, vocab_size]
      W_U_proj_T = W_U_proj.T  # [vocab_size, d_head]

      if prompt_only:
          # Use only tokens from the prompt (first batch)
          prompt_token_ids = tokens[0].tolist()
          # Remove duplicates while preserving order.
          seen = set()
          prompt_token_ids = [tid for tid in prompt_token_ids if tid not in seen and not seen.add(tid)]
          # Optionally filter out the <bos> token.
          if not include_bos and hasattr(self.model.tokenizer, "bos_token_id"):
              bos_id = self.model.tokenizer.bos_token_id
              prompt_token_ids = [tid for tid in prompt_token_ids if tid != bos_id]
          # Build token strings from these prompt token ids.
          if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'decode'):
              token_strs = [self.model.tokenizer.decode([int(tid)]) for tid in prompt_token_ids]
          else:
              token_strs = [f"Token {tid}" for tid in prompt_token_ids]
          # Use these token ids for visualization.
          vis_token_indices = torch.tensor(prompt_token_ids, device=W_U.device)
      else:
          # Build token_strs from the vocabulary subset.
          vocab_size = min(W_U_proj_T.shape[0], self.model.W_E.detach().shape[0])
          token_ids = torch.arange(vocab_size, device=W_U.device)
          if token_strs is None:
              if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'decode'):
                  token_strs = [self.model.tokenizer.decode([int(tid)]) for tid in token_ids.tolist()]
              else:
                  token_strs = [f"Token {i}" for i in range(vocab_size)]
          # Optionally filter out the <bos> token.
          if not include_bos and hasattr(self.model.tokenizer, "bos_token_id"):
              bos_id = self.model.tokenizer.bos_token_id
              # Filter both token_ids and token_strs accordingly.
              filtered = [(tid, s) for tid, s in zip(token_ids.tolist(), token_strs) if tid != bos_id]
              if filtered:
                  token_ids, token_strs = zip(*filtered)
                  token_ids = torch.tensor(token_ids, device=W_U.device)
                  token_strs = list(token_strs)
              else:
                  token_ids = torch.tensor([], device=W_U.device)
                  token_strs = []
          max_tokens = min(top_k * 5, len(token_ids))
          vis_token_indices = token_ids[:max_tokens]

      # Get embeddings for these tokens.
      WE = self.model.W_E.detach()
      if vis_token_indices.numel() > 0 and vis_token_indices.max() >= WE.shape[0]:
          vis_token_indices = vis_token_indices[vis_token_indices < WE.shape[0]]
      sampled_WE = WE[vis_token_indices]

      # Apply MLP0 if available for effective embeddings.
      if hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "mlp"):
          MLP0 = self.model.blocks[0].mlp
          with torch.no_grad():
              effective_WE = MLP0(sampled_WE).detach()
      else:
          effective_WE = sampled_WE

      # Project embeddings through W_O: [tokens, d_head]
      projected_embeddings = torch.matmul(effective_WE, W_O.t())
      sampled_W_U = W_U_proj_T[vis_token_indices]

      # Compute OV circuit matrix.
      ov_matrix = torch.matmul(projected_embeddings, sampled_W_U.t())
      diag_values = torch.diag(ov_matrix)

      # Select top and bottom tokens based on diagonal values.
      top_indices = torch.argsort(diag_values, descending=True)[:min(top_k, len(diag_values))]
      bottom_indices = torch.argsort(diag_values)[:min(top_k, len(diag_values))]
      top_tokens = [token_strs[i] for i in top_indices.tolist()]
      bottom_tokens = [token_strs[i] for i in bottom_indices.tolist()]

      # Plot heatmaps for top and bottom tokens.
      fig, axes = plt.subplots(1, 2, figsize=(55, 55))
      sns.heatmap(
          ov_matrix[top_indices][:, top_indices].cpu().numpy(),
          annot=True, fmt=".2f", cmap="RdBu_r", center=0,
          ax=axes[0], xticklabels=top_tokens, yticklabels=top_tokens
      )
      axes[0].set_title(f"Top {top_k} Tokens with Highest Diagonal Values (L{layer}H{head})")
      axes[0].set_xlabel("Token")
      axes[0].set_ylabel("Token")

      sns.heatmap(
          ov_matrix[bottom_indices][:, bottom_indices].cpu().numpy(),
          annot=True, fmt=".2f", cmap="RdBu_r", center=0,
          ax=axes[1], xticklabels=bottom_tokens, yticklabels=bottom_tokens
      )
      axes[1].set_title(f"Top {top_k} Tokens with Most Negative Diagonal Values (L{layer}H{head})")
      axes[1].set_xlabel("Token")
      axes[1].set_ylabel("Token")

      plt.tight_layout()
      plt.show()

      # Plot histogram of diagonal values.
      plt.figure(figsize=(12, 6))
      plt.hist(diag_values.cpu().numpy(), bins=50, alpha=0.7)
      plt.axvline(x=0, color='r', linestyle='--', label='Zero')
      plt.title(f"Distribution of Diagonal Values in OV Circuit (L{layer}H{head})")
      plt.xlabel("Diagonal Value")
      plt.ylabel("Count")
      plt.legend()
      plt.grid(alpha=0.3)
      plt.show()

      negative_diag_ratio = (diag_values < 0).float().mean().item()
      print(f"Percentage of negative diagonal values: {negative_diag_ratio*100:.2f}%")
      print(f"Mean diagonal value: {diag_values.mean().item():.4f}")
      print(f"Min diagonal value: {diag_values.min().item():.4f}")
      print(f"Max diagonal value: {diag_values.max().item():.4f}")

      return {
          "negative_diag_ratio": negative_diag_ratio,
          "mean_diagonal": diag_values.mean().item(),
          "min_diagonal": diag_values.min().item(),
          "max_diagonal": diag_values.max().item(),
          "top_tokens": top_tokens,
          "bottom_tokens": bottom_tokens
      }

    def component_level_decomposition(self, tokens, layer, head, answer_tokens=None):
        """
        Perform systematic component-level decomposition of a specific head.
        """
        if answer_tokens is None:
            answer_tokens = self.answer_tokens

        if answer_tokens is None:
            raise ValueError("answer_tokens must be provided")

        # Extract all components
        components = self.extract_head_components(tokens, layer, head)

        # Get baseline performance with no ablations
        with torch.no_grad():
            self.model.reset_hooks()  # Clear any existing hooks
            baseline_logits = self.model(tokens)
            baseline_diff = self.logits_to_ave_logit_diff(baseline_logits, answer_tokens)
            print(f"Baseline logit diff: {baseline_diff.item()}")  # Add debugging

        # Define component combinations to test
        component_tests = {
            # Keep only one component
            "only_q": {"ablate_k": True, "ablate_v": True, "ablate_o": True},
            "only_k": {"ablate_q": True, "ablate_v": True, "ablate_o": True},
            "only_v": {"ablate_q": True, "ablate_k": True, "ablate_o": True},
            "only_o": {"ablate_q": True, "ablate_k": True, "ablate_v": True},

            # Ablate only one component
            "no_q": {"ablate_q": True, "ablate_k": False, "ablate_v": False, "ablate_o": False},
            "no_k": {"ablate_q": False, "ablate_k": True, "ablate_v": False, "ablate_o": False},
            "no_v": {"ablate_q": False, "ablate_k": False, "ablate_v": True, "ablate_o": False},
            "no_o": {"ablate_q": False, "ablate_k": False, "ablate_v": False, "ablate_o": True},

            # Component pairs
            "qk_only": {"ablate_q": False, "ablate_k": False, "ablate_v": True, "ablate_o": True},
            "vo_only": {"ablate_q": True, "ablate_k": True, "ablate_v": False, "ablate_o": False},
        }

        results = []

        for test_name, ablation_config in component_tests.items():
            try:
                # Reset hooks between tests
                self.model.reset_hooks()

                # Define more aggressive ablation hook functions
                hooks = []

                # Define hook names based on the model's structure
                q_hook_name = self.get_hook_name(layer, 'q')
                k_hook_name = self.get_hook_name(layer, 'k')
                v_hook_name = self.get_hook_name(layer, 'v')
                z_hook_name = self.get_hook_name(layer, 'z')  # For output ablation

                # Define more aggressive ablation functions
                if ablation_config.get("ablate_q", False):
                    def hook_q(q, hook):
                        # Zero out the specific head's queries
                        if q.dim() >= 3 and q.size(2) > head:
                            q[:, :, head] = 0
                        return q
                    hooks.append((q_hook_name, hook_q))

                if ablation_config.get("ablate_k", False):
                    def hook_k(k, hook):
                        # Zero out the specific head's keys
                        n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', k.shape[2])
                        kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                        if k.dim() >= 3 and k.size(2) > kv_head:
                            k[:, :, kv_head] = 0
                        return k
                    hooks.append((k_hook_name, hook_k))

                if ablation_config.get("ablate_v", False):
                    def hook_v(v, hook):
                        # Zero out the specific head's values
                        n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', v.shape[2])
                        kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                        if v.dim() >= 3 and v.size(2) > kv_head:
                            v[:, :, kv_head] = 0
                        return v
                    hooks.append((v_hook_name, hook_v))

                if ablation_config.get("ablate_o", False):
                    def hook_z(z, hook):
                        # Zero out the specific head's outputs
                        if z.dim() >= 3 and z.size(2) > head:
                            z[:, :, head] = 0
                        return z
                    hooks.append((z_hook_name, hook_z))

                # Add hooks
                for hook_name, hook_fn in hooks:
                    try:
                        self.model.add_hook(hook_name, hook_fn)
                    except Exception as e:
                        print(f"Warning: Could not add hook {hook_name}: {e}")

                # Run model with hooks
                with torch.no_grad():
                    ablated_logits = self.model(tokens)

                    # Calculate logit difference
                    ablated_diff = self.logits_to_ave_logit_diff(ablated_logits, answer_tokens)

                    # Calculate effect
                    effect = baseline_diff - ablated_diff

                    # Add debugging
                    print(f"{test_name}: Ablated diff = {ablated_diff.item()}, Effect = {effect.item()}")

                    results.append({
                        "Test": test_name,
                        "Baseline Logit Diff": baseline_diff.item(),
                        "Ablated Logit Diff": ablated_diff.item(),
                        "Effect": effect.item(),
                        "Normalized Effect": effect.item() / baseline_diff.item() if baseline_diff.item() != 0 else 0
                    })

            except Exception as e:
                print(f"Error evaluating {test_name}: {e}")
                traceback.print_exc()

        # Reset hooks when done
        self.model.reset_hooks()

        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)

        return results_df

    def linear_probe_components(self, tokens, layer, head, task='entity_object_classification'):
        """
        Apply linear probing to specific components of a head to identify
        what information is encoded where.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            task: Classification task type (default: entity_object_classification)

        Returns:
            DataFrame with linear probe results for each component
        """        
        # Extract components
        components = self.extract_head_components(tokens, layer, head)

        # Get token strings for labeling
        token_strs = [self.model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]

        # Define entity/object classification
        if task == 'entity_object_classification':
            # Entities: Character names and pronouns
            entities = ["John", "Mark", "He", "he", "his", "His"]
            # Objects: Items in the scenario
            objects = ["cat", "box", "basket"]

            # Create labels
            labels = []
            for token in token_strs:
                token = token.strip()
                if any(entity in token for entity in entities):
                    labels.append("entity")
                elif any(obj in token for obj in objects):
                    labels.append("object")
                else:
                    labels.append("other")
        else:
            raise ValueError(f"Unsupported task: {task}")

        # Components to probe
        component_keys = ["q", "k", "v", "z"]  # z is the output of attention before W_O

        probe_results = []

        # Probe each component
        for component_key in component_keys:
            if component_key not in components or components[component_key] is None:
                continue

            # Get component values
            component_values = components[component_key][0].detach().cpu().numpy()  # First batch

            # Prepare data
            X = []
            y = []

            for i, label in enumerate(labels):
                if label != "other":  # Skip "other" tokens for cleaner binary classification
                    X.append(component_values[i])
                    y.append(1 if label == "entity" else 0)  # Binary: entity=1, object=0

            if len(set(y)) < 2:
                # Skip if there's only one class
                continue

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train logistic regression
            clf = LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 10),
                cv=min(5, len(X_train)),  # Ensure cv doesn't exceed dataset size
                scoring='balanced_accuracy',
                solver='liblinear',
                class_weight='balanced',
                max_iter=1000
            )

            try:
                clf.fit(X_train, y_train)

                # Evaluate
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

                # Record results
                probe_results.append({
                    "Component": component_key,
                    "Accuracy": accuracy,
                    "Balanced Accuracy": balanced_accuracy,
                    "Best C": clf.C_[0],
                    "Coef Norm": np.linalg.norm(clf.coef_[0])
                })
            except Exception as e:
                print(f"Error probing {component_key}: {e}")

        # Add W_O transformed output if available
        if "z" in components and "W_O" in components and components["z"] is not None and components["W_O"] is not None:
            try:
                # Calculate z·W_O for each token
                z_values = components["z"][0].detach()  # First batch
                W_O = components["W_O"]

                transformed_output = torch.matmul(z_values, W_O).cpu().numpy()

                # Prepare data
                X = []
                y = []

                for i, label in enumerate(labels):
                    if label != "other":  # Skip "other" tokens
                        X.append(transformed_output[i])
                        y.append(1 if label == "entity" else 0)

                if len(set(y)) >= 2:
                    # Convert to numpy arrays
                    X = np.array(X)
                    y = np.array(y)

                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Train logistic regression
                    clf = LogisticRegressionCV(
                        Cs=np.logspace(-4, 4, 10),
                        cv=min(5, len(X_train)),
                        scoring='balanced_accuracy',
                        solver='liblinear',
                        class_weight='balanced',
                        max_iter=1000
                    )

                    clf.fit(X_train, y_train)

                    # Evaluate
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

                    # Record results
                    probe_results.append({
                        "Component": "z·W_O",
                        "Accuracy": accuracy,
                        "Balanced Accuracy": balanced_accuracy,
                        "Best C": clf.C_[0],
                        "Coef Norm": np.linalg.norm(clf.coef_[0])
                    })
            except Exception as e:
                print(f"Error probing z·W_O: {e}")

        return pd.DataFrame(probe_results)

    def component_interaction_analysis(self, tokens, layer, head, answer_tokens=None):
        """
        Analyze the interaction between different components (Q, K, V, O)
        """
        # Use the already improved component_level_decomposition
        ablation_results = self.component_level_decomposition(tokens, layer, head, answer_tokens)

        # Extract individual component effects
        individual_effects = {}
        for comp in ["q", "k", "v", "o"]:
            test_name = f"no_{comp}"
            if test_name in ablation_results["Test"].values:
                row = ablation_results[ablation_results["Test"] == test_name].iloc[0]
                individual_effects[comp] = row["Effect"]

        # Analyze pairs
        component_pairs = [
            ("q", "k"),
            ("q", "v"),
            ("q", "o"),
            ("k", "v"),
            ("k", "o"),
            ("v", "o")
        ]

        results = []

        # For each pair, look for existing pair tests or run new ablations
        for comp1, comp2 in component_pairs:
            # Try to find existing pair test
            pair_name = None
            if comp1 == "q" and comp2 == "k" or comp1 == "k" and comp2 == "q":
                # Find the qk_only test and compute the effect from other components
                pair_name = "vo_only"  # This ablates q and k, so the effect is from v and o
            elif comp1 == "v" and comp2 == "o" or comp1 == "o" and comp2 == "v":
                pair_name = "qk_only"  # This ablates v and o, so the effect is from q and k

            if pair_name and pair_name in ablation_results["Test"].values:
                # Calculate pair effect from the opposite pair
                row = ablation_results[ablation_results["Test"] == pair_name].iloc[0]
                base_diff = row["Baseline Logit Diff"]
                # The effect of comp1+comp2 is the baseline minus the effect of the other components
                pair_effect = base_diff - row["Ablated Logit Diff"]
            else:
                # We don't have a direct test for this pair, so calculate from individual effects
                pair_effect = individual_effects.get(comp1, 0) + individual_effects.get(comp2, 0)

            # Calculate expected effect if components were independent (sum of individual effects)
            expected_effect = individual_effects.get(comp1, 0) + individual_effects.get(comp2, 0)

            # Calculate interaction effect (deviation from additivity)
            interaction_effect = pair_effect - expected_effect

            results.append({
                "Component Pair": f"{comp1}-{comp2}",
                "Pair Effect": pair_effect,
                "Sum of Individual Effects": expected_effect,
                "Interaction Effect": interaction_effect,
                "Normalized Interaction": interaction_effect / abs(expected_effect) if expected_effect != 0 else 0
            })

        return pd.DataFrame(results)

    def visualize_component_decomposition(self, tokens, layer, head, include_token_analysis=True):
        """
        Create comprehensive visualizations of component-level decomposition
        to better understand the role of each component in the head's function.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            include_token_analysis: Whether to include token-level breakdowns

        Returns:
            Dictionary with visualization figures
        """        
        # Extract components
        components = self.extract_head_components(tokens, layer, head)

        # Get token strings for analysis
        token_strs = [self.model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]

        # Figures to return
        figures = {}

        # 1. PCA visualization of different component representations
        pca = PCA(n_components=2)

        component_pcas = {}
        for comp_name, comp_data in components.items():
            if comp_name in ["q", "k", "v", "z"] and comp_data is not None:
                try:
                    # Get first batch
                    comp_values = comp_data[0].detach().cpu().numpy()

                    # Apply PCA
                    pca_result = pca.fit_transform(comp_values)
                    component_pcas[comp_name] = pca_result
                except Exception as e:
                    print(f"Error processing {comp_name} for PCA: {e}")

        # Plot PCA results
        if component_pcas:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[f"{comp.upper()} Component" for comp in component_pcas.keys()]
            )

            row, col = 1, 1
            for comp_name, pca_result in component_pcas.items():
                scatter = go.Scatter(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    mode='markers+text',
                    text=token_strs,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=list(range(len(pca_result))),
                        colorscale='Viridis',
                    ),
                    name=f"{comp_name.upper()} Component"
                )

                fig.add_trace(scatter, row=row, col=col)

                col += 1
                if col > 2:
                    col = 1
                    row += 1

            fig.update_layout(
                height=800,
                width=1000,
                title_text=f"Component PCA for L{layer}H{head}"
            )

            figures["component_pca"] = fig

        # 2. Analysis of diagonal values in component contributions to OV circuit
        if "v" in components and "W_O" in components and components["v"] is not None and components["W_O"] is not None:
            try:
                # Get first batch of v
                v_values = components["v"][0].detach()
                W_O = components["W_O"]

                # Compute output contribution
                output_contribution = torch.matmul(v_values, W_O)

                # Get diagonal values
                if include_token_analysis:
                    # Create dictionary to categorize tokens
                    token_categories = {}

                    # Simple categorization
                    entities = ["John", "Mark", "He", "he"]
                    objects = ["cat", "box", "basket"]

                    for i, token in enumerate(token_strs):
                        token = token.strip()
                        if any(entity in token for entity in entities):
                            category = "entity"
                        elif any(obj in token for obj in objects):
                            category = "object"
                        elif token in [".", ",", "the", "a", "in", "on", "and", "is", "to", "of"]:
                            category = "syntax"
                        else:
                            category = "other"

                        if category not in token_categories:
                            token_categories[category] = []

                        token_categories[category].append((i, token))

                    # Create visualization of value contributions by token category
                    fig = go.Figure()

                    colors = {
                        "entity": "blue",
                        "object": "green",
                        "syntax": "red",
                        "other": "gray"
                    }

                    for category, tokens in token_categories.items():
                        if tokens:
                            indices = [t[0] for t in tokens]
                            values = output_contribution[indices].mean(dim=1).cpu().numpy()

                            fig.add_trace(go.Box(
                                y=values,
                                name=category,
                                boxpoints='all',
                                jitter=0.3,
                                marker=dict(color=colors[category]),
                                pointpos=0,
                                text=[t[1] for t in tokens]
                            ))

                    fig.update_layout(
                        title=f"Output Contribution by Token Category for L{layer}H{head}",
                        yaxis_title="Mean Output Contribution",
                        boxmode='group'
                    )

                    figures["token_category_contribution"] = fig

            except Exception as e:
                print(f"Error analyzing component contributions: {e}")

        return figures

    # Use with new component analysis code
    # def debug_component_analysis(self, tokens, layer, head, answer_tokens=None):
    #     print(f"Debugging component analysis for L{layer}H{head}...")
    #     print("\n1. Verifying model and tokens:")
    #     print(f"Model type: {type(self.model)}")
    #     print(f"Token shape: {tokens.shape}")
    #     print(f"Sample tokens: {tokens[0, :5]}")
    #     print("\n2. Extracting components:")
    #     components = self.extract_head_components(tokens, layer, head)
    #     for comp_name, comp_data in components.items():
    #         if comp_name in ["q", "k", "v", "z", "W_O"]:
    #             if comp_data is not None:
    #                 print(f"{comp_name} shape: {comp_data.shape}")
    #             else:
    #                 print(f"{comp_name} is None")
    #     print("\n3. Testing basic ablation:")
    #     self.model.reset_hooks()
    #     try:
    #         with torch.no_grad():
    #             baseline_logits = self.model(tokens)
    #             if answer_tokens is not None:
    #                 baseline_diff = self.sequence_logit_diff(tokens, answer_tokens[:,0], answer_tokens[:,1])
    #                 print(f"Baseline logit diff: {baseline_diff.item()}")
    #             else:
    #                 print("No answer tokens provided for logit diff calculation")
    #         print("Testing ablation of q component:")
    #         q_hook_name = self.get_hook_name(layer, 'q')
    #         def hook_q(q, hook):
    #             if q.dim() >= 3 and q.size(2) > head:
    #                 orig_values = q[:, :, head].clone()
    #                 q[:, :, head] = 0
    #                 print(f"Ablated q values from {orig_values.mean().item()} to {q[:, :, head].mean().item()}")
    #             return q
    #         self.model.add_hook(q_hook_name, hook_q)
    #         with torch.no_grad():
    #             ablated_logits = self.model(tokens)
    #             if answer_tokens is not None:
    #                 ablated_diff = self.sequence_logit_diff(tokens, answer_tokens[:,0], answer_tokens[:,1])
    #                 print(f"Ablated logit diff: {ablated_diff.item()}")
    #                 print(f"Effect: {(baseline_diff - ablated_diff).item()}")
    #             else:
    #                 print("No answer tokens provided for logit diff calculation")
    #         self.model.reset_hooks()
    #     except Exception as e:
    #         print(f"Error in ablation test: {e}")
    #         import traceback
    #         traceback.print_exc()
    #     if answer_tokens is not None:
    #         print("\n4. Verifying answer tokens:")
    #         print(f"Answer tokens shape: {answer_tokens.shape}")
    #         print(f"Answer tokens: {answer_tokens}")
    #         try:
    #             answers = [self.model.to_string(answer_tokens[0, i:i+1]) for i in range(answer_tokens.shape[1])]
    #             print(f"Answer strings: {answers}")
    #         except Exception as e:
    #             print(f"Error converting answer tokens to strings: {e}")
    #     print("\n5. Testing logit calculation:")
    #     try:
    #         with torch.no_grad():
    #             logits = self.model(tokens)
    #             print(f"Logits shape: {logits.shape}")
    #             final_logits = logits[:, -1, :]
    #             print(f"Final logits shape: {final_logits.shape}")
    #             if answer_tokens is not None:
    #                 answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    #                 print(f"Answer logits shape: {answer_logits.shape}")
    #                 print(f"Answer logits: {answer_logits}")
    #                 if answer_logits.shape[1] >= 2:
    #                     logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    #                     print(f"Logit difference: {logit_diff.item()}")
    #     except Exception as e:
    #         print(f"Error in logit calculation test: {e}")
    #         import traceback
    #         traceback.print_exc()
    #     print("\nDebugging complete.")
    #     return

    # Use with negative diagonal analysis code
    def debug_component_analysis(self, tokens, layer, head, answer_tokens=None):
        """
        Debug the component analysis to identify where issues might be occurring.
        """
        print(f"Debugging component analysis for L{layer}H{head}...")

        # 1. Verify model and tokens
        print("\n1. Verifying model and tokens:")
        print(f"Model type: {type(self.model)}")
        print(f"Token shape: {tokens.shape}")
        print(f"Sample tokens: {tokens[0, :5]}")

        # 2. Extract and verify components
        print("\n2. Extracting components:")
        components = self.extract_head_components(tokens, layer, head)

        for comp_name, comp_data in components.items():
            if comp_name in ["q", "k", "v", "z", "W_O"]:
                if comp_data is not None:
                    print(f"{comp_name} shape: {comp_data.shape}")
                else:
                    print(f"{comp_name} is None")

        # 3. Test basic ablation
        print("\n3. Testing basic ablation:")
        self.model.reset_hooks()

        try:
            # Run baseline
            with torch.no_grad():
                baseline_logits = self.model(tokens)
                if answer_tokens is not None:
                    baseline_diff = self.logits_to_ave_logit_diff(baseline_logits, answer_tokens)
                    print(f"Baseline logit diff: {baseline_diff.item()}")
                else:
                    print("No answer tokens provided for logit diff calculation")

            # Try a single component ablation
            print("Testing ablation of q component:")
            q_hook_name = self.get_hook_name(layer, 'q')

            def hook_q(q, hook):
                # Zero out the specific head's queries
                if q.dim() >= 3 and q.size(2) > head:
                    orig_values = q[:, :, head].clone()
                    q[:, :, head] = 0
                    print(f"Ablated q values from {orig_values.mean().item()} to {q[:, :, head].mean().item()}")
                return q

            self.model.add_hook(q_hook_name, hook_q)

            with torch.no_grad():
                ablated_logits = self.model(tokens)
                if answer_tokens is not None:
                    ablated_diff = self.logits_to_ave_logit_diff(ablated_logits, answer_tokens)
                    print(f"Ablated logit diff: {ablated_diff.item()}")
                    print(f"Effect: {(baseline_diff - ablated_diff).item()}")
                else:
                    print("No answer tokens provided for logit diff calculation")

            self.model.reset_hooks()
        except Exception as e:
            print(f"Error in ablation test: {e}")
            traceback.print_exc()

        # 4. Verify answer tokens
        if answer_tokens is not None:
            print("\n4. Verifying answer tokens:")
            print(f"Answer tokens shape: {answer_tokens.shape}")
            print(f"Answer tokens: {answer_tokens}")

            # Try to convert tokens to strings
            try:
                answers = [self.model.to_string(answer_tokens[0, i:i+1]) for i in range(answer_tokens.shape[1])]
                print(f"Answer strings: {answers}")
            except Exception as e:
                print(f"Error converting answer tokens to strings: {e}")

        # 5. Test logit calculation
        print("\n5. Testing logit calculation:")
        try:
            with torch.no_grad():
                logits = self.model(tokens)
                print(f"Logits shape: {logits.shape}")

                # Get final logits
                final_logits = logits[:, -1, :]
                print(f"Final logits shape: {final_logits.shape}")

                if answer_tokens is not None:
                    # Get logits for answer tokens
                    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
                    print(f"Answer logits shape: {answer_logits.shape}")
                    print(f"Answer logits: {answer_logits}")

                    # Calculate difference
                    if answer_logits.shape[1] >= 2:
                        logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
                        print(f"Logit difference: {logit_diff.item()}")
        except Exception as e:
            print(f"Error in logit calculation test: {e}")
            traceback.print_exc()

        print("\nDebugging complete.")
        return

    def extend_analyzer_with_eigenvalue_methods(CircuitAnalyzer):
        """
        Extend the CircuitAnalyzer class with eigenvalue decomposition methods.
        This uses Python's class modification at runtime.

        Args:
            CircuitAnalyzer: The class to extend

        Returns:
            The modified class
        """
        # Add all the methods from OVCircuitAnalyzer
        setattr(CircuitAnalyzer, 'compute_ov_matrix', compute_ov_matrix)
        setattr(CircuitAnalyzer, 'analyze_eigenvalue_spectrum', analyze_eigenvalue_spectrum)
        setattr(CircuitAnalyzer, 'visualize_eigenvalue_spectrum', visualize_eigenvalue_spectrum)
        setattr(CircuitAnalyzer, 'compare_eigenvalue_distributions', compare_eigenvalue_distributions)
        setattr(CircuitAnalyzer, 'track_eigenvalue_flow', track_eigenvalue_flow)
        setattr(CircuitAnalyzer, 'analyze_token_representation_dimensionality', analyze_token_representation_dimensionality)

        return CircuitAnalyzer

    def run_component_decomposition_analysis(self, tokens, layer, head, answer_tokens=None):
        """
        Run a comprehensive component-level decomposition analysis
        for a given head, combining all the individual analysis methods.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            answer_tokens: Token IDs for answers

        Returns:
            Dictionary with all analysis results
        """
        print(f"Running comprehensive component decomposition analysis for L{layer}H{head}...")

        results = {}

        # 1. Component-level ablation study
        print("Performing component-level ablation study...")
        try:
            ablation_results = self.component_level_decomposition(tokens, layer, head, answer_tokens)
            results["ablation_study"] = ablation_results
            print("Component ablation study completed.")
        except Exception as e:
            print(f"Error in component ablation study: {e}")

        # 2. Linear probing of components
        print("Performing linear probe analysis...")
        try:
            probe_results = self.linear_probe_components(tokens, layer, head)
            results["linear_probe"] = probe_results
            print("Linear probe analysis completed.")
        except Exception as e:
            print(f"Error in linear probe analysis: {e}")

        # 3. Component interaction analysis
        print("Analyzing component interactions...")
        try:
            interaction_results = self.component_interaction_analysis(tokens, layer, head, answer_tokens)
            results["interaction_analysis"] = interaction_results
            print("Component interaction analysis completed.")
        except Exception as e:
            print(f"Error in component interaction analysis: {e}")

        # 4. Visualization of component decomposition
        print("Creating visualizations...")
        try:
            vis_results = self.visualize_component_decomposition(tokens, layer, head)
            results["visualizations"] = vis_results
            print("Visualizations created.")
        except Exception as e:
            print(f"Error creating visualizations: {e}")

        print(f"Component decomposition analysis for L{layer}H{head} completed.")
        return results

    def compute_ov_matrix(self, tokens, layer, head, use_effective_embeddings=True, vocab_subset_size=2000):
        """
        Compute the full OV circuit matrix for a head.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            use_effective_embeddings: Whether to use MLP0 transformed embeddings
            vocab_subset_size: Number of tokens to sample (for efficiency)

        Returns:
            OV circuit matrix of shape [vocab_size, vocab_size] or [subset_size, subset_size]
        """
        # Extract components
        components = self.extract_head_components(tokens, layer, head)

        # Check if we have the necessary components
        if "W_O" not in components or components["W_O"] is None:
            raise ValueError("W_O component not available")

        W_O = components["W_O"]

        # Get unembedding matrix
        W_U = self.model.W_U.detach()  # [d_model, vocab_size]

        # Compute projected unembedding: [d_head, vocab_size]
        W_U_proj = torch.matmul(W_O, W_U)
        W_U_proj_T = W_U_proj.T  # [vocab_size, d_head]

        # Get embedding matrix
        WE = self.model.W_E.detach()  # [vocab_size, d_model]

        # Sample tokens for efficiency if needed
        vocab_size = WE.shape[0]
        if vocab_subset_size and vocab_subset_size < vocab_size:
            # Sample random indices
            indices = torch.randperm(vocab_size, device=WE.device)[:vocab_subset_size]
        else:
            # Use all tokens
            indices = torch.arange(vocab_size, device=WE.device)
            vocab_subset_size = vocab_size

        # Apply MLP0 if requested and available (for effective embeddings)
        if use_effective_embeddings and hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "mlp"):
            MLP0 = self.model.blocks[0].mlp
            # Process embeddings in batches to avoid OOM
            effective_WE = []
            batch_size = 100
            sampled_WE = WE[indices]
            for i in range(0, len(indices), batch_size):
                end_idx = min(i + batch_size, len(indices))
                batch = sampled_WE[i:end_idx]
                with torch.no_grad():
                    effective_batch = MLP0(batch)
                effective_WE.append(effective_batch.detach())
            effective_WE = torch.cat(effective_WE, dim=0)
        else:
            effective_WE = WE[indices]

        # Project embeddings through W_O: [subset_size, d_head]
        projected_embeddings = torch.matmul(effective_WE, W_O.t())

        # Compute OV circuit matrix for the subset: [subset_size, subset_size]
        sampled_W_U = W_U_proj_T[indices]
        ov_matrix = torch.matmul(projected_embeddings, sampled_W_U.t())

        return ov_matrix

    def analyze_eigenvalue_spectrum(self, ov_matrix, k=None):
        """
        Perform eigenvalue decomposition on the OV matrix.

        Args:
            ov_matrix: OV circuit matrix [vocab_size, vocab_size] or [subset_size, subset_size]
            k: Number of eigenvalues to compute (None for all)

        Returns:
            Dictionary with eigenvalues, eigenvectors, and statistics
        """        
        # Convert to numpy for computation
        ov_matrix_np = ov_matrix.cpu().numpy()

        # For large matrices, compute only the top k eigenvalues
        if k is not None and k < ov_matrix_np.shape[0]:
            # Use scipy's sparse eigenvalue computation            
            try:
                eigenvalues, eigenvectors = eigs(ov_matrix_np, k=k)
            except Exception as e:
                print(f"Warning: Sparse eigendecomposition failed, falling back to full: {e}")
                eigenvalues, eigenvectors = linalg.eig(ov_matrix_np)
        else:
            # Full eigendecomposition
            eigenvalues, eigenvectors = linalg.eig(ov_matrix_np)

        # Sort eigenvalues by magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute statistics
        real_eigenvalues = np.real(eigenvalues)
        imag_eigenvalues = np.imag(eigenvalues)

        negative_ratio = (real_eigenvalues < 0).mean()
        real_mean = np.mean(real_eigenvalues)
        real_std = np.std(real_eigenvalues)
        abs_mean = np.mean(np.abs(eigenvalues))

        # Calculate effective dimensionality (participation entropy)
        normalized_evals = np.abs(eigenvalues) / np.sum(np.abs(eigenvalues))
        effective_dim = np.exp(-np.sum(normalized_evals * np.log(normalized_evals + 1e-10)))

        # Calculate participation ratio
        participation_ratio = np.sum(np.abs(eigenvalues))**2 / np.sum(np.abs(eigenvalues)**2)

        # Calculate other metrics
        dim_90pct = np.argmax(np.cumsum(normalized_evals) >= 0.9) + 1 if np.any(np.cumsum(normalized_evals) >= 0.9) else len(normalized_evals)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "negative_ratio": negative_ratio,
            "real_mean": real_mean,
            "real_std": real_std,
            "abs_mean": abs_mean,
            "effective_dim": effective_dim,
            "participation_ratio": participation_ratio,
            "dim_90pct": dim_90pct,
            "top_eigenvalue": eigenvalues[0],
            "eigenvalue_decay": np.abs(eigenvalues[0]) / (np.abs(eigenvalues[-1]) + 1e-10) if len(eigenvalues) > 1 else 1.0
        }

    def visualize_eigenvalue_spectrum(self, eigenvalue_data, title=None):
        """
        Visualize the eigenvalue spectrum.

        Args:
            eigenvalue_data: Result from analyze_eigenvalue_spectrum
            title: Plot title

        Returns:
            Plotly figure
        """        
        eigenvalues = eigenvalue_data["eigenvalues"]
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        magnitudes = np.abs(eigenvalues)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Eigenvalue Magnitudes",
                "Eigenvalues in Complex Plane",
                "Real Parts Distribution",
                "Decay Curve (log scale)"
            ]
        )

        # Plot 1: Eigenvalue magnitudes
        fig.add_trace(
            go.Bar(
                x=list(range(len(magnitudes))),
                y=magnitudes,
                name="Magnitude"
            ),
            row=1, col=1
        )

        # Plot 2: Eigenvalues in complex plane
        fig.add_trace(
            go.Scatter(
                x=real_parts,
                y=imag_parts,
                mode="markers",
                marker=dict(
                    size=5,
                    color=np.log1p(magnitudes),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Log Magnitude")
                ),
                name="Complex Eigenvalues"
            ),
            row=1, col=2
        )

        # Plot 3: Real parts distribution
        fig.add_trace(
            go.Histogram(
                x=real_parts,
                nbinsx=30,
                name="Real Parts"
            ),
            row=2, col=1
        )

        # Plot 4: Decay curve
        fig.add_trace(
            go.Scatter(
                x=list(range(len(magnitudes))),
                y=magnitudes,
                mode="lines",
                name="Magnitude"
            ),
            row=2, col=2
        )

        # Update y-axis of decay curve to log scale
        fig.update_yaxes(type="log", row=2, col=2)

        # Add key statistics as annotations
        annotations = [
            f"Negative ratio: {eigenvalue_data['negative_ratio']:.2f}",
            f"Effective dim: {eigenvalue_data['effective_dim']:.2f}",
            f"Participation ratio: {eigenvalue_data['participation_ratio']:.2f}",
            f"Dim 90%: {eigenvalue_data['dim_90pct']}",
            f"Top eigenvalue: {np.abs(eigenvalue_data['top_eigenvalue']):.2f}"
        ]

        for i, annotation in enumerate(annotations):
            fig.add_annotation(
                x=0.5, y=0.05 - i*0.025,
                xref="paper", yref="paper",
                text=annotation,
                showarrow=False,
                font=dict(size=10)
            )

        # Update layout
        if title is None:
            title = "Eigenvalue Spectrum Analysis"

        fig.update_layout(
            title=title,
            height=800,
            width=1000,
            showlegend=False
        )

        return fig

    def compare_eigenvalue_distributions(self, tokens, head_specs, use_effective_embeddings=True, k=100):
        """
        Compare eigenvalue distributions across multiple heads or models.

        Args:
            tokens: Input tokens
            head_specs: List of (layer, head) tuples to compare
            use_effective_embeddings: Whether to use MLP0 transformed embeddings
            k: Number of top eigenvalues to compare

        Returns:
            Comparison visualizations and statistics
        """        
        all_results = []

        for layer, head in head_specs:
            try:
                # Compute OV matrix
                ov_matrix = self.compute_ov_matrix(tokens, layer, head, use_effective_embeddings)

                # Analyze eigenvalue spectrum
                spectrum_data = self.analyze_eigenvalue_spectrum(ov_matrix, k=k)

                # Store results
                all_results.append({
                    "layer": layer,
                    "head": head,
                    "spectrum_data": spectrum_data,
                    "name": f"L{layer}H{head}"
                })
            except Exception as e:
                print(f"Error analyzing L{layer}H{head}: {e}")

        if not all_results:
            return None

        # Create comparison visualizations

        # 1. Eigenvalue magnitudes comparison
        fig1 = go.Figure()

        for result in all_results:
            magnitudes = np.abs(result["spectrum_data"]["eigenvalues"])[:k]
            name = result["name"]

            fig1.add_trace(
                go.Scatter(
                    x=list(range(len(magnitudes))),
                    y=magnitudes,
                    mode="lines",
                    name=name
                )
            )

        fig1.update_layout(
            title="Eigenvalue Magnitude Comparison",
            xaxis_title="Eigenvalue Index",
            yaxis_title="Magnitude",
            yaxis_type="log",
            height=600,
            width=800
        )

        # 2. Negative ratio and effective dimensionality comparison
        labels = [result["name"] for result in all_results]
        negative_ratios = [result["spectrum_data"]["negative_ratio"] for result in all_results]
        effective_dims = [result["spectrum_data"]["effective_dim"] for result in all_results]
        participation_ratios = [result["spectrum_data"]["participation_ratio"] for result in all_results]

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=labels,
            y=negative_ratios,
            name="Negative Ratio",
            text=[f"{nr:.2f}" for nr in negative_ratios],
            textposition="auto"
        ))

        fig2.add_trace(go.Bar(
            x=labels,
            y=[ed/max(effective_dims) for ed in effective_dims],  # Normalize
            name="Effective Dim (norm)",
            text=[f"{ed:.2f}" for ed in effective_dims],
            textposition="auto"
        ))

        fig2.add_trace(go.Bar(
            x=labels,
            y=[pr/max(participation_ratios) for pr in participation_ratios],  # Normalize
            name="Participation Ratio (norm)",
            text=[f"{pr:.2f}" for pr in participation_ratios],
            textposition="auto"
        ))

        fig2.update_layout(
            title="Eigenvalue Distribution Metrics Comparison",
            xaxis_title="Head",
            yaxis_title="Value",
            barmode="group",
            height=500,
            width=800
        )

        # Compile comparison statistics
        statistics = pd.DataFrame([
            {
                "Name": result["name"],
                "Negative Ratio": result["spectrum_data"]["negative_ratio"],
                "Effective Dim": result["spectrum_data"]["effective_dim"],
                "Participation Ratio": result["spectrum_data"]["participation_ratio"],
                "Dim 90%": result["spectrum_data"]["dim_90pct"],
                "Top Eigenvalue": np.abs(result["spectrum_data"]["top_eigenvalue"]),
                "Eigenvalue Decay": result["spectrum_data"]["eigenvalue_decay"],
                "Real Mean": result["spectrum_data"]["real_mean"]
            }
            for result in all_results
        ])

        return {
            "figures": {
                "magnitude_comparison": fig1,
                "metrics_comparison": fig2
            },
            "statistics": statistics,
            "raw_results": all_results
        }

    def track_eigenvalue_flow(self, tokens, head, layers, use_effective_embeddings=True, k=100):
        """
        Track how eigenvalue properties change across layers for a specific head position.

        Args:
            tokens: Input tokens
            head: Head index (position)
            layers: List of layers to analyze
            use_effective_embeddings: Whether to use MLP0 transformed embeddings
            k: Number of top eigenvalues to compute

        Returns:
            Visualization and statistics of eigenvalue flow across layers
        """        
        layer_results = []

        for layer in layers:
            try:
                # Compute OV matrix
                ov_matrix = self.compute_ov_matrix(tokens, layer, head, use_effective_embeddings)

                # Analyze eigenvalue spectrum
                spectrum_data = self.analyze_eigenvalue_spectrum(ov_matrix, k=k)

                # Store results
                layer_results.append({
                    "layer": layer,
                    "spectrum_data": spectrum_data
                })
            except Exception as e:
                print(f"Error analyzing layer {layer}: {e}")

        if not layer_results:
            return None

        # Create flow visualizations

        # Track changes in key metrics across layers
        layers = [result["layer"] for result in layer_results]
        negative_ratios = [result["spectrum_data"]["negative_ratio"] for result in layer_results]
        effective_dims = [result["spectrum_data"]["effective_dim"] for result in layer_results]
        participation_ratios = [result["spectrum_data"]["participation_ratio"] for result in layer_results]
        top_eigenvalues = [np.abs(result["spectrum_data"]["top_eigenvalue"]) for result in layer_results]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=negative_ratios,
                mode="lines+markers",
                name="Negative Ratio"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=[ed/max(effective_dims) for ed in effective_dims],  # Normalize for scale
                mode="lines+markers",
                name="Effective Dim (normalized)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=[pr/max(participation_ratios) for pr in participation_ratios],  # Normalize
                mode="lines+markers",
                name="Participation Ratio (normalized)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=[te/max(top_eigenvalues) for te in top_eigenvalues],  # Normalize for scale
                mode="lines+markers",
                name="Top Eigenvalue (normalized)"
            )
        )

        fig.update_layout(
            title=f"Eigenvalue Properties Flow Across Layers (Head Position {head})",
            xaxis_title="Layer",
            yaxis_title="Normalized Value",
            height=600,
            width=800
        )

        # Compile flow statistics
        statistics = pd.DataFrame([
            {
                "Layer": result["layer"],
                "Negative Ratio": result["spectrum_data"]["negative_ratio"],
                "Effective Dim": result["spectrum_data"]["effective_dim"],
                "Participation Ratio": result["spectrum_data"]["participation_ratio"],
                "Dim 90%": result["spectrum_data"]["dim_90pct"],
                "Top Eigenvalue": np.abs(result["spectrum_data"]["top_eigenvalue"]),
                "Eigenvalue Decay": result["spectrum_data"]["eigenvalue_decay"]
            }
            for result in layer_results
        ])

        return {
            "figure": fig,
            "statistics": statistics,
            "raw_results": layer_results
        }

    def analyze_token_representation_dimensionality(self, tokens, layer, head, token_strs=None):
        """
        Analyze the dimensionality of token representations by category.

        Args:
            tokens: Input tokens
            layer: Layer index
            head: Head index
            token_strs: Token strings for categorization

        Returns:
            Analysis of token representation dimensionality by category
        """        
        # Extract components
        components = self.extract_head_components(tokens, layer, head)

        # Generate token strings if not provided
        if token_strs is None:
            try:
                token_strs = [self.model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]
            except Exception as e:
                print(f"Could not generate token strings: {e}")
                token_strs = [f"Token {i}" for i in range(tokens.shape[1])]

        # Categorize tokens
        token_categories = {}

        # Define categories (adjust based on specific context for the task) Adjust as needed
        entities = ["John", "Mark", "He", "he", "his", "His"]
        objects = ["cat", "box", "basket"]
        syntax = [".", ",", "the", "a", "in", "on", "and", "is", "to", "of"]

        for i, token in enumerate(token_strs):
            token = token.strip()
            if any(entity in token for entity in entities):
                category = "entity"
            elif any(obj in token for obj in objects):
                category = "object"
            elif any(syn in token for syn in syntax):
                category = "syntax"
            else:
                category = "other"

            if category not in token_categories:
                token_categories[category] = []

            token_categories[category].append(i)

        results = {}

        # Analyze different component representations
        for comp_name in ["q", "k", "v", "z"]:
            if comp_name not in components or components[comp_name] is None:
                continue

            comp_values = components[comp_name][0].detach().cpu().numpy()

            category_results = {}

            for category, indices in token_categories.items():
                if not indices or len(indices) < 2:  # Need at least 2 samples for PCA
                    continue

                # Extract values for this category
                category_values = comp_values[indices]

                # Compute PCA
                pca = PCA().fit(category_values)

                # Compute effective dimensionality based on PCA variance
                explained_variance_ratio = pca.explained_variance_ratio_
                normalized_ratios = explained_variance_ratio / np.sum(explained_variance_ratio)
                entropy_dim = np.exp(-np.sum(normalized_ratios * np.log(normalized_ratios + 1e-10)))

                # Participation ratio
                participation_ratio = np.sum(explained_variance_ratio)**2 / np.sum(explained_variance_ratio**2)

                # Find dimensionality for 90% explained variance
                cumulative_var = np.cumsum(explained_variance_ratio)
                dim_90pct = np.argmax(cumulative_var >= 0.9) + 1 if np.any(cumulative_var >= 0.9) else len(cumulative_var)

                category_results[category] = {
                    "effective_dim": entropy_dim,
                    "participation_ratio": participation_ratio,
                    "dim_90pct": dim_90pct,
                    "explained_variance_ratio": explained_variance_ratio,
                    "cumulative_variance": cumulative_var
                }

            results[comp_name] = category_results

        # Create visualizations

        # 1. Effective dimensionality comparison across components and categories
        fig1 = go.Figure()
        comp_cat_labels = []
        effective_dims = []
        participation_ratios = []

        for comp_name, comp_results in results.items():
            for category, cat_results in comp_results.items():
                comp_cat_labels.append(f"{comp_name}-{category}")
                effective_dims.append(cat_results["effective_dim"])
                participation_ratios.append(cat_results["participation_ratio"])

        fig1.add_trace(
            go.Bar(
                x=comp_cat_labels,
                y=effective_dims,
                name="Effective Dim"
            )
        )

        fig1.add_trace(
            go.Bar(
                x=comp_cat_labels,
                y=participation_ratios,
                name="Participation Ratio"
            )
        )

        fig1.update_layout(
            title=f"Token Representation Dimensionality (L{layer}H{head})",
            xaxis_title="Component-Category",
            yaxis_title="Dimensionality Measure",
            barmode="group",
            height=600,
            width=1000
        )

        # 2. Variance curves for each component and category
        fig2 = make_subplots(
            rows=len(results),
            cols=1,
            subplot_titles=[f"{comp_name.upper()} Component" for comp_name in results]
        )

        row = 1
        for comp_name, comp_results in results.items():
            for category, cat_results in comp_results.items():
                fig2.add_trace(
                    go.Scatter(
                        x=list(range(1, len(cat_results["cumulative_variance"]) + 1)),
                        y=cat_results["cumulative_variance"],
                        mode="lines",
                        name=f"{category}"
                    ),
                    row=row, col=1
                )

            # Add 90% variance horizontal line
            fig2.add_shape(
                type="line",
                x0=0,
                y0=0.9,
                x1=max(len(cat_results["cumulative_variance"]) for cat_results in comp_results.values()),
                y1=0.9,
                line=dict(color="red", dash="dash"),
                row=row, col=1
            )

            row += 1

        fig2.update_layout(
            title=f"Cumulative Variance by Component and Token Category (L{layer}H{head})",
            height=800,
            width=1000
        )

        # Compile statistics
        statistics = []

        for comp_name, comp_results in results.items():
            for category, cat_results in comp_results.items():
                statistics.append({
                    "Component": comp_name,
                    "Category": category,
                    "Effective Dim": cat_results["effective_dim"],
                    "Participation Ratio": cat_results["participation_ratio"],
                    "Dim 90%": cat_results["dim_90pct"]
                })

        statistics_df = pd.DataFrame(statistics)

        return {
            "figures": {
                "dimensionality": fig1,
                "variance_curves": fig2
            },
            "statistics": statistics_df,
            "raw_results": results
        }

    # Method to add to the CircuitAnalyzer class
    def run_cross_model_eigenvalue_comparison(self, model_specs=None, use_relative_positions=False, relative_positions=None):
        """
        Compare eigenvalue distributions across different model architectures.

        Args:
            model_specs: Dictionary mapping model names to lists of (layer, head, label) tuples
                        If None, will use relative positions or default specs
            use_relative_positions: Whether to use relative positions instead of fixed layer/head indices
            relative_positions: Dictionary of relative positions and labels
                              Default is early (25%), middle (50%), late (75%)

        Returns:
            DataFrame with comparison results and visualizations
        """
        # Create a prompt if not already available (use the one from the analyzer)
        prompt = getattr(self, 'dataset', None) or "In the room there are John, Mark, a cat, a box, and a basket. John takes the cat and puts it on the basket. He leaves the room and goes to school. While John is away, Mark takes the cat off the basket and puts it on the box. Mark leaves the room and goes to work. John comes back from school and enters the room. John looks around the room. He doesn't know what happened in the room when he was away. John thinks the cat is on the"

        # Define default model specs if none provided
        if model_specs is None:
            if use_relative_positions:
                model_specs = self._generate_relative_position_specs(relative_positions)
            else:
                # Default to some common models with reasonable defaults
                model_specs = {
                    "EleutherAI/gpt-neo-1.3B": [
                        (10, 8, "Late Layer, Late Head"),
                        (22, 9, "Late Layer, Late, Head")
                    ],
                    "EleutherAI/pythia-1.4b": [
                        (16, 10, "Late Layer, Late Head"),
                        (28, 15, "Late Layer, Late Head")
                    ],
                    "facebook/opt-1.3b": [
                        (12, 6, "Late Layer, Late Head"),
                        (20, 8, "Late Layer, Late Head")
                    ]
                }

        # Store analysis results
        results = []

        # Keep track of figure objects to return
        figures = {}

        # Analyze each model
        for model_name, head_specs in model_specs.items():
            print(f"Analyzing {model_name}...")
            try:
                # Load model
                model = HookedTransformer.from_pretrained(model_name)
                tokens = model.to_tokens(prompt, prepend_bos=True)

                # Create a temporary analyzer for this model
                # We'll use the same class as the current analyzer
                temp_analyzer = self.__class__(model, prompt)

                # Add eigenvalue analysis methods if not already present
                if not hasattr(temp_analyzer, 'compute_ov_matrix'):
                    # from eigenvalue_analysis import extend_analyzer_with_eigenvalue_methods
                    temp_analyzer = extend_analyzer_with_eigenvalue_methods(temp_analyzer)

                # For each specified head in this model
                for layer, head, label in head_specs:
                    # Ensure layer and head are valid for this model
                    layer = min(layer, model.cfg.n_layers - 1)
                    head = min(head, model.cfg.n_heads - 1)

                    print(f"  Analyzing {model_name} L{layer}H{head} ({label})...")

                    # Compute OV matrix
                    ov_matrix = temp_analyzer.compute_ov_matrix(tokens, layer, head, vocab_subset_size=1000)

                    # Analyze eigenvalue spectrum
                    spectrum = temp_analyzer.analyze_eigenvalue_spectrum(ov_matrix)

                    # Store results
                    results.append({
                        "model": model_name,
                        "model_short": model_name.split("/")[-1],
                        "layer": layer,
                        "head": head,
                        "label": label,
                        "negative_ratio": spectrum["negative_ratio"],
                        "effective_dim": spectrum["effective_dim"],
                        "participation_ratio": spectrum["participation_ratio"],
                        "top_eigenvalue": np.abs(spectrum["top_eigenvalue"]),
                        "dim_90pct": spectrum["dim_90pct"],
                        "eigenvalue_decay": spectrum["eigenvalue_decay"],
                        "real_mean": spectrum["real_mean"],
                        "spectrum": spectrum
                    })

                    # Visualize individual spectrum
                    fig = temp_analyzer.visualize_eigenvalue_spectrum(
                        spectrum,
                        f"Eigenvalue Spectrum for {model_name.split('/')[-1]} L{layer}H{head} ({label})"
                    )
                    figures[f"{model_name}_L{layer}H{head}"] = fig

                # Clean up to free memory
                del temp_analyzer
                del model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error analyzing {model_name}: {e}")
                traceback.print_exc()

        # Create comparison dataframe
        if not results:
            print("No results to display.")
            return None, {}

        comparison_df = pd.DataFrame([
            {k: v for k, v in result.items() if k != "spectrum"}
            for result in results
        ])

        # Create comparison visualizations

        # 1. Compare negative ratios across models for same functional head types
        labels = sorted(set(r["label"] for r in results))
        fig1 = go.Figure()

        for label in labels:
            label_data = [r for r in results if r["label"] == label]
            model_names = [r["model_short"] for r in label_data]
            negative_ratios = [r["negative_ratio"] for r in label_data]

            fig1.add_trace(go.Bar(
                x=model_names,
                y=negative_ratios,
                name=label,
                text=[f"{nr:.2f}" for nr in negative_ratios],
                textposition="auto"
            ))

        fig1.update_layout(
            title="Negative Eigenvalue Ratio by Model and Head Type",
            xaxis_title="Model",
            yaxis_title="Negative Ratio",
            barmode="group",
            height=500,
            width=800
        )
        figures["negative_ratio_comparison"] = fig1

        # 2. Compare effective dimensionality
        fig2 = go.Figure()

        for label in labels:
            label_data = [r for r in results if r["label"] == label]
            model_names = [r["model_short"] for r in label_data]
            effective_dims = [r["effective_dim"] for r in label_data]

            fig2.add_trace(go.Bar(
                x=model_names,
                y=effective_dims,
                name=label,
                text=[f"{ed:.2f}" for ed in effective_dims],
                textposition="auto"
            ))

        fig2.update_layout(
            title="Effective Dimensionality by Model and Head Type",
            xaxis_title="Model",
            yaxis_title="Effective Dimensionality",
            barmode="group",
            height=500,
            width=800
        )
        figures["effective_dim_comparison"] = fig2

        # 3. Eigenvalue decay curves
        fig3 = make_subplots(
            rows=len(labels),
            cols=1,
            subplot_titles=[f"Eigenvalue Decay: {label}" for label in labels]
        )

        for i, label in enumerate(labels):
            label_data = [r for r in results if r["label"] == label]

            for result in label_data:
                eigenvalues = np.abs(result["spectrum"]["eigenvalues"])
                normalized_eigenvalues = eigenvalues / eigenvalues[0]  # Normalize

                fig3.add_trace(
                    go.Scatter(
                        x=list(range(min(50, len(normalized_eigenvalues)))),
                        y=normalized_eigenvalues[:50],  # Show first 50 eigenvalues
                        mode="lines",
                        name=f"{result['model_short']}"
                    ),
                    row=i+1, col=1
                )

            # Use log scale for y-axis
            fig3.update_yaxes(type="log", row=i+1, col=1)

        fig3.update_layout(
            height=300 * len(labels),
            width=800,
            showlegend=(len(labels) == 1)  # Only show legend for single subplot
        )
        figures["eigenvalue_decay_comparison"] = fig3

        # 4. Radar chart for architecture comparison
        if len(results) >= 3:  # Only create radar chart if we have enough data
            # Group by model and calculate averages
            model_stats = {}

            for result in results:
                model = result["model_short"]
                if model not in model_stats:
                    model_stats[model] = {
                        "negative_ratio": [],
                        "effective_dim": [],
                        "participation_ratio": [],
                        "top_eigenvalue": [],
                        "eigenvalue_decay": []
                    }

                model_stats[model]["negative_ratio"].append(result["negative_ratio"])
                model_stats[model]["effective_dim"].append(result["effective_dim"])
                model_stats[model]["participation_ratio"].append(result["participation_ratio"])
                model_stats[model]["top_eigenvalue"].append(result["top_eigenvalue"])
                model_stats[model]["eigenvalue_decay"].append(result["eigenvalue_decay"])

            # Calculate averages
            for model in model_stats:
                for metric in model_stats[model]:
                    model_stats[model][metric] = np.mean(model_stats[model][metric])

            # Create radar chart
            fig4 = go.Figure()

            categories = ["Negative Ratio", "Effective Dim", "Participation Ratio",
                        "Top Eigenvalue", "Eigenvalue Decay"]

            # Normalize values for radar chart
            max_values = {
                "negative_ratio": max(stat["negative_ratio"] for stat in model_stats.values()),
                "effective_dim": max(stat["effective_dim"] for stat in model_stats.values()),
                "participation_ratio": max(stat["participation_ratio"] for stat in model_stats.values()),
                "top_eigenvalue": max(stat["top_eigenvalue"] for stat in model_stats.values()),
                "eigenvalue_decay": max(stat["eigenvalue_decay"] for stat in model_stats.values())
            }

            for model_name, stats in model_stats.items():
                normalized_stats = [
                    stats["negative_ratio"] / max_values["negative_ratio"],
                    stats["effective_dim"] / max_values["effective_dim"],
                    stats["participation_ratio"] / max_values["participation_ratio"],
                    stats["top_eigenvalue"] / max_values["top_eigenvalue"],
                    stats["eigenvalue_decay"] / max_values["eigenvalue_decay"]
                ]

                fig4.add_trace(go.Scatterpolar(
                    r=normalized_stats,
                    theta=categories,
                    fill='toself',
                    name=model_name
                ))

            fig4.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Architecture Signatures",
                height=600,
                width=800
            )

            figures["architecture_signature"] = fig4

        return comparison_df, figures

    def _generate_relative_position_specs(self, relative_positions=None):
        """
        Helper method to generate model specs based on relative positions in each model.

        Args:
            relative_positions: Dictionary of position name -> (fraction, label)
                              If None, uses defaults

        Returns:
            Dictionary of model specs with appropriate layer indices
        """
        if relative_positions is None:
            # Default to analyzing early, middle and late layers
            relative_positions = {
                "early": (0.25, "Early Layer Head"),
                "middle": (0.5, "Middle Layer Head"),
                "late": (0.75, "Late Layer Head")
            }

        # Define models to analyze
        models_to_analyze = [
            "mistral-7b",
            "meta-llama/Meta-Llama-3-8B",
            "EleutherAI/pythia-1.4b-v0"
        ]

        # Build model specs dictionary
        model_specs = {}

        for model_name in models_to_analyze:
            try:
                # Load model just to get architecture info
                print(f"Loading {model_name} to determine architecture...")
                model = HookedTransformer.from_pretrained(model_name)

                # Get number of layers and heads
                n_layers = model.cfg.n_layers
                n_heads = model.cfg.n_heads

                # Default to head 0 or middle head
                default_head = min(n_heads // 2, n_heads - 1)

                # Create specifications for this model
                model_specs[model_name] = []

                for position_name, (fraction, label) in relative_positions.items():
                    # Calculate layer index based on fraction of model depth
                    layer_idx = min(int(n_layers * fraction), n_layers - 1)

                    # Add to model specs
                    model_specs[model_name].append((layer_idx, default_head, f"{label} ({position_name})"))

                # Clean up model to free memory
                del model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error determining architecture for {model_name}: {e}")

        return model_specs

    def weights_based_verification(self, tokens=None, layer=None, head=None, threshold=0.75,
                              visualize=True, sample_size=2000):
        """
        Perform direct weights-based verification of suppression mechanisms by analyzing
        the OV and QK circuits without relying solely on activations.

        This follows the methodology in the copy suppression paper where they analyze
        diagonal elements of WUWLOV and other weight-based patterns.

        Args:
            tokens: Optional input tokens (for context-sensitive analysis)
            layer: Layer to analyze (if None, uses previously specified layer)
            head: Head to analyze (if None, uses previously specified head)
            threshold: Threshold for classifying suppression behavior (default: 0.75)
            visualize: Whether to create visualizations (default: True)
            sample_size: Number of tokens to sample for efficiency (default: 2000)

        Returns:
            Dictionary with analysis results and metrics
        """        
        # If layer/head not provided, try to use instance variables
        if layer is None or head is None:
            raise ValueError("Layer and head must be specified")

        # Adjust layer and head to be valid for the model
        layer, head = self.get_valid_layer_head(layer, head)
        print(f"Analyzing weights for L{layer}H{head}...")

        # Extract components (even if tokens not provided, can get weights)
        components = {}
        if tokens is not None:
            components = self.extract_head_components(tokens, layer, head)

        # Get W_O directly from model if not in components
        if "W_O" not in components or components["W_O"] is None:
            try:
                # Get W_O directly from model architecture
                if hasattr(self.model, 'blocks'):
                    W_O = self.model.blocks[layer].attn.W_O[head].detach()
                elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    # Approximate for GPT-2 style models
                    W_O = self.model.transformer.h[layer].attn.c_proj.weight.view(
                        self.model.cfg.n_heads, self.model.cfg.d_head, self.model.cfg.d_model
                    )[head].detach()
                else:
                    raise ValueError("Could not access W_O directly from model architecture")
            except Exception as e:
                print(f"Error accessing W_O directly: {e}")
                W_O = None
        else:
            W_O = components["W_O"]

        if W_O is None:
            raise ValueError("Could not retrieve W_O weight matrix")

        # Get embedding and unembedding matrices
        W_E = self.model.W_E.detach()
        W_U = self.model.W_U.detach()

        # Get MLP0 if available for effective embeddings
        has_mlp0 = False
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "mlp"):
            MLP0 = self.model.blocks[0].mlp
            has_mlp0 = True

        # Sample tokens for efficiency
        vocab_size = min(W_U.shape[1], W_E.shape[0])
        max_tokens = min(sample_size, vocab_size)
        token_indices = torch.randperm(vocab_size)[:max_tokens].to(W_U.device)

        # Use MLP0 output as effective embedding if available (as in paper)
        if has_mlp0:
            print("Using MLP0 output as effective embedding...")
            batched_indices = token_indices.split(100)  # Process in batches to avoid OOM
            effective_WE_list = []
            for indices_batch in batched_indices:
                with torch.no_grad():
                    batch_embeddings = W_E[indices_batch]
                    batch_effective = MLP0(batch_embeddings)
                    effective_WE_list.append(batch_effective)
            effective_WE = torch.cat(effective_WE_list, dim=0)
        else:
            effective_WE = W_E[token_indices]

        # Get dimensions
        d_head = W_O.shape[0] if W_O.ndim == 2 else W_O.shape[1]
        d_model = W_O.shape[1] if W_O.ndim == 2 else W_O.shape[2]

        # Check for architecture with different head and model dimensions
        # (like Llama which has d_head=128 and d_model=4096)
        different_dims = d_head != d_model

        # Compute OV circuit: effective_WE → W_O → W_U
        try:
            if different_dims:
                # For architectures with different dimensions, we need to handle this differently
                # Project embeddings through the head's output matrix
                # effective_WE: [batch, d_model]
                # W_O: [d_head, d_model]
                # First create head-sized projections
                head_dim_projections = torch.zeros((effective_WE.shape[0], d_head), device=effective_WE.device)

                # Project them through W_O (we need to iterate by batch for large models)
                batch_size = 100
                projected_embeddings_list = []
                for i in range(0, effective_WE.shape[0], batch_size):
                    batch_end = min(i + batch_size, effective_WE.shape[0])
                    batch = effective_WE[i:batch_end]
                    # Project to model dimension
                    if W_O.ndim == 2:  # [d_head, d_model]
                        proj_batch = torch.matmul(head_dim_projections[i:batch_end], W_O)
                    else:  # [head, d_head, d_model]
                        proj_batch = torch.matmul(head_dim_projections[i:batch_end], W_O[0])
                    projected_embeddings_list.append(proj_batch)

                projected_embeddings = torch.cat(projected_embeddings_list, dim=0)

                # Now project through unembedding
                W_U_projected = W_U[:, token_indices]
                ov_circuit = torch.matmul(projected_embeddings, W_U_projected)
            else:
                # Standard approach for models with same dimensions
                if W_O.ndim == 2:  # [d_head, d_model]
                    projected_embeddings = torch.matmul(effective_WE, W_O.t())
                else:  # [head, d_head, d_model]
                    projected_embeddings = torch.matmul(effective_WE, W_O[0].t())

                W_U_projected = W_U[:, token_indices]
                ov_circuit = torch.matmul(projected_embeddings, W_U_projected)

            # Extract diagonal elements (key for copy suppression analysis)
            diag_values = torch.diag(ov_circuit)

            # Compute metrics used in the paper
            diag_negative_ratio = (diag_values < 0).float().mean().item()

            # Calculate rank metrics (what rank are the diagonal elements in their columns/rows?)
            bottom_5_percent_count = 0
            top_10_negative_count = 0
            diagonal_ranks = []

            for i in range(ov_circuit.shape[0]):
                col = ov_circuit[:, i]
                # Sort column values from lowest to highest
                sorted_col, _ = torch.sort(col)
                # Find rank of diagonal element (how many elements are smaller)
                diag_val = ov_circuit[i, i]
                rank = torch.sum(sorted_col < diag_val).item()
                normalized_rank = rank / len(col)
                diagonal_ranks.append(normalized_rank)

                # Check if diagonal is in bottom 5%
                threshold_idx = max(1, int(0.05 * len(col)))
                if torch.topk(col, threshold_idx, largest=False)[0][-1] >= diag_val:
                    bottom_5_percent_count += 1

                # Check if diagonal is in top 10 negative values
                if torch.topk(col, min(10, len(col)), largest=False)[0][-1] >= diag_val:
                    top_10_negative_count += 1

            bottom_5_percent_ratio = bottom_5_percent_count / len(token_indices)
            top_10_negative_ratio = top_10_negative_count / len(token_indices)
            median_rank = np.median(diagonal_ranks)
        except Exception as e:
            print(f"Error computing OV circuit: {e}")
            traceback.print_exc()
            # Return basic results with default values
            return {
                "layer": layer,
                "head": head,
                "diag_negative_ratio": 0.0,
                "bottom_5_percent_ratio": 0.0,
                "top_10_negative_ratio": 0.0,
                "median_rank": 0.5,
                "suppression_score": 0.0,
                "is_suppression_head": False,
                "error": str(e)
            }

        # Try computing QK circuit (this is optional, so code continues even if it fails)
        try:
            # Get W_Q and W_K
            qk_diag_max_ratio = 0.0
            qk_diag_max_ratio_2 = 0.0

            # Skip this for now as it's less critical and more complex to fix
            # If needed, implement similar dimensionality handling here
        except Exception as e:
            print(f"QK circuit computation skipped: {e}")
            qk_diag_max_ratio = 0.0
            qk_diag_max_ratio_2 = 0.0

        # Combine metrics to determine suppression behavior
        suppression_score = (
            0.25 * diag_negative_ratio +
            0.25 * bottom_5_percent_ratio +
            0.25 * top_10_negative_ratio +
            0.25 * (1 - median_rank)  # Lower rank is better for suppression
        )

        # Classification based on paper thresholds
        is_suppression_head = suppression_score > threshold

        # Create visualizations
        plots = {}
        if visualize:
            try:
                # 1. Histogram of diagonal values
                fig = plt.figure(figsize=(10, 6))
                plt.hist(diag_values.cpu().numpy(), bins=30, alpha=0.7)
                plt.axvline(x=0, color='r', linestyle='--', label='Zero')
                plt.title(f"Distribution of Diagonal Values in OV Circuit (L{layer}H{head})")
                plt.xlabel("Value")
                plt.ylabel("Count")
                plt.legend()
                plt.grid(alpha=0.3)
                plots["diag_histogram"] = fig
                plt.close(fig)

                # 2. Heatmap of OV circuit with diagonal highlighted
                # Sample a subset for visualization
                sample_size = min(50, ov_circuit.shape[0])
                sample_indices = torch.randperm(ov_circuit.shape[0])[:sample_size]
                ov_sample = ov_circuit[sample_indices][:, sample_indices].cpu().numpy()

                fig = px.imshow(
                    ov_sample,
                    title=f"OV Circuit Matrix (L{layer}H{head}) - Sample",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0
                )
                # Highlight diagonal
                for i in range(sample_size):
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5, y0=i-0.5, y1=i+0.5,
                        line=dict(color="black", width=2),
                        fillcolor="rgba(0,0,0,0)"
                    )
                plots["ov_matrix"] = fig

                # 3. Scatterplot of diagonal ranks
                fig = px.histogram(
                    diagonal_ranks,
                    title=f"Distribution of Diagonal Element Ranks (L{layer}H{head})",
                    labels={"value": "Normalized Rank", "count": "Frequency"},
                    opacity=0.7
                )
                fig.add_vline(x=0.05, line_dash="dash", line_color="red", annotation_text="Bottom 5%")
                plots["rank_distribution"] = fig
            except Exception as e:
                print(f"Error creating visualizations: {e}")

        # Return results
        return {
            "layer": layer,
            "head": head,
            "diag_negative_ratio": diag_negative_ratio,
            "bottom_5_percent_ratio": bottom_5_percent_ratio,
            "top_10_negative_ratio": top_10_negative_ratio,
            "median_rank": median_rank,
            "qk_diag_max_ratio": qk_diag_max_ratio,
            "suppression_score": suppression_score,
            "is_suppression_head": is_suppression_head,
            "ov_circuit_sample": ov_circuit[:10, :10].cpu().numpy() if visualize else None,
            "plots": plots if visualize else None
        }

    def perpendicular_component_analysis(self, tokens, layer, head, target_token_idx=-1,
                                        unembedding_tokens=None, visualize=True):
        """
        Analyze the importance of directions perpendicular to unembedding vectors,
        following the methodology in the copy suppression paper section 4.2.

        The paper found that suppression mechanisms rely on more than just the
        unembedding direction, and perpendicular components are crucial.

        Args:
            tokens: Input tokens
            layer: Layer to analyze
            head: Head to analyze
            target_token_idx: Position to analyze (default: last token)
            unembedding_tokens: Tokens to use for unembedding directions (if None, derived from context)
            visualize: Whether to create visualizations

        Returns:
            Dictionary with analysis results including parallel vs perpendicular importance
        """       
        # Adjust layer and head to be valid for the model
        layer, head = self.get_valid_layer_head(layer, head)

        # Run the model and cache activations
        components = self.extract_head_components(tokens, layer, head, return_cache=True)
        cache = components.get("cache", None)

        if cache is None:
            raise ValueError("Could not extract activation cache")

        # Get token strings for better interpretability
        try:
            token_strs = [self.model.to_string(tokens[0, i:i+1]) for i in range(tokens.shape[1])]
        except:
            token_strs = [f"Token_{i}" for i in range(tokens.shape[1])]

        # If unembedding_tokens not specified, derive from context
        if unembedding_tokens is None:
            # For simplicity, use the top predicted tokens at the target position
            with torch.no_grad():
                logits = self.model(tokens)
                if target_token_idx < 0:
                    target_token_idx = tokens.shape[1] + target_token_idx
                top_logits, top_tokens = logits[0, target_token_idx].topk(5)
                unembedding_tokens = top_tokens

        # Get unembedding directions for these tokens
        W_U = self.model.W_U.detach()
        if isinstance(unembedding_tokens, torch.Tensor):
            unembedding_vectors = W_U[:, unembedding_tokens].t()  # [n_tokens, d_model]
        else:
            unembedding_vectors = None
            print("Could not extract unembedding vectors")

        # Results storage
        results = {
            "layer": layer,
            "head": head,
            "parallel_importance": {},
            "perpendicular_importance": {},
            "combined_metrics": {},
            "attention_shifts": {},
            "plots": {}
        }

        try:
            # Extract query and key activations
            query = None
            key = None

            # Find query and key in cache
            for hook_name in cache:
                if "hook_q" in hook_name and f"{layer}" in hook_name:
                    query_full = cache[hook_name]
                    # Extract head-specific query
                    if query_full.ndim == 4:  # [batch, seq, head, d_head]
                        query = query_full[0, :, head]
                    elif query_full.ndim == 3 and query_full.shape[1] == self.model.cfg.n_heads:  # [batch, head, d_head or seq*d_head]
                        query = query_full[0, head]
                    elif query_full.ndim == 3:  # [batch, seq, head*d_head]
                        q_reshape = query_full.reshape(query_full.shape[0], query_full.shape[1], self.model.cfg.n_heads, -1)
                        query = q_reshape[0, :, head]

                if "hook_k" in hook_name and f"{layer}" in hook_name:
                    key_full = cache[hook_name]
                    # Extract head-specific key, accounting for KV sharing
                    n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', key_full.shape[2] if key_full.ndim == 4 else self.model.cfg.n_heads)
                    kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)

                    if key_full.ndim == 4:  # [batch, seq, head, d_head]
                        key = key_full[0, :, kv_head]
                    elif key_full.ndim == 3 and key_full.shape[1] == n_kv_heads:  # [batch, head, d_head or seq*d_head]
                        key = key_full[0, kv_head]
                    elif key_full.ndim == 3:  # [batch, seq, head*d_head]
                        k_reshape = key_full.reshape(key_full.shape[0], key_full.shape[1], n_kv_heads, -1)
                        key = k_reshape[0, :, kv_head]

            if query is None or key is None:
                print("Could not extract query and key activations")
                results["error"] = "Could not extract query and key activations"
                return results

            # Extract pattern (attention weights) if available
            pattern = None
            for hook_name in cache:
                if "hook_pattern" in hook_name and f"{layer}" in hook_name:
                    pattern_full = cache[hook_name]
                    if pattern_full.ndim == 4:  # [batch, head, q_seq, k_seq]
                        pattern = pattern_full[0, head]
                    elif pattern_full.ndim == 3:  # [batch, q_seq, k_seq]
                        pattern = pattern_full[0]

            # For models with different head and model dimensions (like Llama)
            query_dim = query.shape[-1]
            model_dim = W_U.shape[0]
            have_different_dims = query_dim != model_dim

            # 1. Analyze parallel and perpendicular components at target position
            target_query = query[target_token_idx]

            # Analyze for each unembedding direction
            for i, token_id in enumerate(unembedding_tokens):
                if i >= 5:  # Limit to top 5 tokens
                    break

                token_str = self.model.to_string([token_id])

                # Get unembedding vector
                if unembedding_vectors is not None:
                    unembed_vec = unembedding_vectors[i]

                    # Handle dimension mismatch for models like Llama
                    if have_different_dims:
                        # Create a projection matrix or adapter for the query vector
                        if hasattr(self.model, 'blocks') and hasattr(self.model.blocks[layer].attn, 'W_O'):
                            # Use W_O to project from head dimension to model dimension
                            W_O = self.model.blocks[layer].attn.W_O[head].detach()
                            # Project query to model dimension
                            projected_query = torch.matmul(target_query.unsqueeze(0), W_O).squeeze(0)
                        else:
                            # Fallback: create a simple mapping (less accurate)
                            projected_query = torch.zeros(model_dim, device=target_query.device)
                            # Copy values where possible
                            min_dim = min(query_dim, model_dim)
                            projected_query[:min_dim] = target_query[:min_dim]

                        # Now use the projected query for analysis
                        query_to_use = projected_query
                    else:
                        query_to_use = target_query

                    # Normalize for projection calculations
                    unembed_norm = torch.norm(unembed_vec)
                    if unembed_norm > 0:
                        unembed_unit = unembed_vec / unembed_norm

                        # Project query onto unembedding direction
                        parallel_component = torch.dot(query_to_use, unembed_unit) * unembed_unit
                        perpendicular_component = query_to_use - parallel_component

                        # Calculate magnitudes
                        parallel_mag = torch.norm(parallel_component).item()
                        perpendicular_mag = torch.norm(perpendicular_component).item()
                        total_mag = torch.norm(query_to_use).item()

                        # Store importance metrics
                        results["parallel_importance"][token_str] = {
                            "magnitude": parallel_mag,
                            "proportion": (parallel_mag / total_mag) if total_mag > 0 else 0
                        }

                        results["perpendicular_importance"][token_str] = {
                            "magnitude": perpendicular_mag,
                            "proportion": (perpendicular_mag / total_mag) if total_mag > 0 else 0
                        }

                # Skip intervention analysis as it's complex to adapt for different architectures

            # 3. Aggregated metrics
            if results["parallel_importance"] and results["perpendicular_importance"]:
                avg_parallel_proportion = np.mean([v["proportion"] for v in results["parallel_importance"].values()])
                avg_perpendicular_proportion = np.mean([v["proportion"] for v in results["perpendicular_importance"].values()])

                results["combined_metrics"] = {
                    "avg_parallel_proportion": avg_parallel_proportion,
                    "avg_perpendicular_proportion": avg_perpendicular_proportion,
                    "perpendicular_dominance": avg_perpendicular_proportion > avg_parallel_proportion,
                    "parallel_perpendicular_ratio": avg_parallel_proportion / avg_perpendicular_proportion if avg_perpendicular_proportion > 0 else float('inf')
                }

            # Skip visualizations for simplicity in this fixed version
            # Add back if needed

        except Exception as e:
            print(f"Error in perpendicular component analysis: {e}")
            traceback.print_exc()
            results["error"] = str(e)

        return results

    def analyze_ov_negative_diagonal_patterns(self, tokens=None, layer=None, head=None,
                                         token_category_fn=None, sample_size=2000,
                                         category_labels=None, visualize=True):
        """
        Analyze negative diagonal patterns in the OV circuit across different token types,
        specifically focusing on OV suppression mechanisms as described in the copy
        suppression paper.

        Args:
            tokens: Optional input tokens
            layer: Layer to analyze
            head: Head to analyze
            token_category_fn: Function that takes a token string and returns a category label
                              (if None, uses a default categorization)
            sample_size: Number of tokens to sample
            category_labels: Labels for token categories (if None, derived from data)
            visualize: Whether to create visualizations

        Returns:
            Dictionary with analysis of negative diagonal patterns by token category
        """        
        # Default token categorization function
        if token_category_fn is None:
            def default_categorize(token_str):
                token_str = token_str.strip() if hasattr(token_str, 'strip') else str(token_str).strip()
                token_lower = token_str.lower()

                # Common function words
                function_words = ["the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "but", "for", "with", "by", "as"]
                punctuation = [".", ",", "!", "?", "\"", "'", ";", ":"]

                if any(token_lower == fw for fw in function_words):
                    return "function_word"
                elif any(p in token_str for p in punctuation):
                    return "punctuation"
                elif token_str.isdigit() or (token_str.startswith('-') and token_str[1:].isdigit()):
                    return "number"
                elif token_lower[0:1].isalpha() and token_lower[0:1].upper() == token_str[0:1]:
                    return "capitalized"
                elif token_str.isupper():
                    return "uppercase"
                else:
                    return "content_word"

            token_category_fn = default_categorize

        # If layer/head not provided, try to use instance variables or default
        if layer is None or head is None:
            raise ValueError("Layer and head must be specified")

        # Adjust layer and head to be valid for the model
        layer, head = self.get_valid_layer_head(layer, head)
        print(f"Analyzing OV negative diagonal patterns for L{layer}H{head}...")

        # Get components - W_O directly from model
        try:
            if hasattr(self.model, 'blocks'):
                W_O = self.model.blocks[layer].attn.W_O[head].detach()
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                W_O = self.model.transformer.h[layer].attn.c_proj.weight.view(
                    self.model.cfg.n_heads, self.model.cfg.d_head, self.model.cfg.d_model
                )[head].detach()
            else:
                raise ValueError("Could not access W_O directly from model architecture")
        except Exception as e:
            print(f"Error accessing W_O: {e}")
            return {
                "layer": layer,
                "head": head,
                "error": f"Could not access W_O weights: {str(e)}"
            }

        # Get embedding and unembedding matrices
        W_E = self.model.W_E.detach()
        W_U = self.model.W_U.detach()

        # Get MLP0 if available for effective embeddings
        has_mlp0 = False
        try:
            if hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "mlp"):
                MLP0 = self.model.blocks[0].mlp
                has_mlp0 = True
        except:
            has_mlp0 = False

        # Sample tokens for efficiency
        vocab_size = min(W_U.shape[1], W_E.shape[0])
        max_tokens = min(sample_size, vocab_size)
        token_indices = torch.randperm(vocab_size)[:max_tokens].to(W_U.device)

        # Convert indices to token strings for categorization
        token_strs = []
        for idx in token_indices.cpu().tolist():
            try:
                token_str = self.model.to_string([idx])
                token_strs.append(token_str)
            except:
                token_strs.append(f"Token_{idx}")

        # Categorize tokens
        token_categories = [token_category_fn(token_str) for token_str in token_strs]

        # Check for dimension mismatch
        d_head = W_O.shape[0] if W_O.ndim == 2 else W_O.shape[1]
        d_model = W_O.shape[1] if W_O.ndim == 2 else W_O.shape[2]
        different_dims = d_head != d_model

        # Use MLP0 output as effective embedding if available
        try:
            if has_mlp0:
                print("Using MLP0 output as effective embedding...")
                batched_indices = token_indices.split(100)  # Process in batches to avoid OOM
                effective_WE_list = []
                for indices_batch in batched_indices:
                    with torch.no_grad():
                        batch_embeddings = W_E[indices_batch]
                        batch_effective = MLP0(batch_embeddings)
                        effective_WE_list.append(batch_effective)
                effective_WE = torch.cat(effective_WE_list, dim=0)
            else:
                effective_WE = W_E[token_indices]

            # Compute OV circuit
            if different_dims:
                # For models with different dimensions, create an intermediate step
                head_dim_projections = torch.zeros((effective_WE.shape[0], d_head), device=effective_WE.device)

                # Project through W_O (we need to iterate by batch for large models)
                batch_size = 100
                projected_embeddings_list = []
                for i in range(0, effective_WE.shape[0], batch_size):
                    batch_end = min(i + batch_size, effective_WE.shape[0])
                    batch = effective_WE[i:batch_end]
                    # Project to model dimension
                    if W_O.ndim == 2:  # [d_head, d_model]
                        proj_batch = torch.matmul(head_dim_projections[i:batch_end], W_O)
                    else:  # [head, d_head, d_model]
                        proj_batch = torch.matmul(head_dim_projections[i:batch_end], W_O[0])
                    projected_embeddings_list.append(proj_batch)

                projected_embeddings = torch.cat(projected_embeddings_list, dim=0)
            else:
                # Standard approach for models with same dimensions
                if W_O.ndim == 2:  # [d_head, d_model]
                    projected_embeddings = torch.matmul(effective_WE, W_O.t())
                else:  # [head, d_head, d_model]
                    projected_embeddings = torch.matmul(effective_WE, W_O[0].t())

            # Apply unembedding
            W_U_projected = W_U[:, token_indices]
            ov_circuit = torch.matmul(projected_embeddings, W_U_projected)

            # Extract diagonal elements
            diag_values = torch.diag(ov_circuit).cpu().numpy()
        except Exception as e:
            print(f"Error computing OV circuit: {e}")
            traceback.print_exc()
            # Return error
            return {
                "layer": layer,
                "head": head,
                "error": f"Error computing OV circuit: {str(e)}"
            }

        # Create a dataframe for analysis
        df = pd.DataFrame({
            'token': token_strs,
            'token_id': token_indices.cpu().numpy(),
            'category': token_categories,
            'diagonal_value': diag_values,
            'is_negative': diag_values < 0
        })

        # Calculate metrics by category
        category_metrics = df.groupby('category').agg({
            'token': 'count',
            'diagonal_value': ['mean', 'std', 'min', 'max'],
            'is_negative': 'mean'
        }).reset_index()

        # Rename for clarity
        category_metrics.columns = ['category', 'count', 'diag_mean', 'diag_std', 'diag_min', 'diag_max', 'negative_ratio']

        # Calculate ranks of diagonal elements
        try:
            diagonal_ranks = []
            bottom_5_percent_count = 0
            bottom_10_percent_count = 0

            for i in range(ov_circuit.shape[0]):
                col = ov_circuit[:, i]
                sorted_col, _ = torch.sort(col)
                diag_val = ov_circuit[i, i]

                # Find rank (how many values are smaller than diagonal)
                rank = torch.sum(sorted_col < diag_val).item()
                normalized_rank = rank / len(col)
                diagonal_ranks.append(normalized_rank)

                # Check if in bottom 5% and 10%
                bottom_5_pct = max(1, int(0.05 * len(col)))
                bottom_10_pct = max(1, int(0.1 * len(col)))

                if torch.topk(col, bottom_5_pct, largest=False)[0][-1] >= diag_val:
                    bottom_5_percent_count += 1

                if torch.topk(col, bottom_10_pct, largest=False)[0][-1] >= diag_val:
                    bottom_10_percent_count += 1

            # Add ranks to dataframe
            df['normalized_rank'] = diagonal_ranks
            df['in_bottom_5_percent'] = [1 if r < 0.05 else 0 for r in diagonal_ranks]
            df['in_bottom_10_percent'] = [1 if r < 0.1 else 0 for r in diagonal_ranks]
        except Exception as e:
            print(f"Error calculating ranks: {e}")
            # Set default values
            df['normalized_rank'] = 0.5
            df['in_bottom_5_percent'] = 0
            df['in_bottom_10_percent'] = 0
            diagonal_ranks = [0.5] * len(df)
            bottom_5_percent_count = 0
            bottom_10_percent_count = 0

        # Update category metrics
        try:
            rank_metrics = df.groupby('category').agg({
                'normalized_rank': ['mean', 'median'],
                'in_bottom_5_percent': 'mean',
                'in_bottom_10_percent': 'mean'
            }).reset_index()

            # Rename columns
            rank_metrics.columns = ['category', 'rank_mean', 'rank_median', 'bottom_5_pct_ratio', 'bottom_10_pct_ratio']

            # Merge metrics
            category_metrics = category_metrics.merge(rank_metrics, on='category')
        except Exception as e:
            print(f"Error calculating category rank metrics: {e}")
            # We'll still return the basic metrics

        # Calculate overall metrics
        overall_metrics = {
            'total_tokens': len(df),
            'diag_negative_ratio': df['is_negative'].mean(),
            'diag_mean': df['diagonal_value'].mean(),
            'diag_std': df['diagonal_value'].std(),
            'bottom_5_percent_ratio': bottom_5_percent_count / len(df) if len(df) > 0 else 0,
            'bottom_10_percent_ratio': bottom_10_percent_count / len(df) if len(df) > 0 else 0,
            'rank_mean': np.mean(diagonal_ranks) if diagonal_ranks else 0.5,
            'rank_median': np.median(diagonal_ranks) if diagonal_ranks else 0.5
        }

        # Create visualizations
        plots = {}
        if visualize:
            try:
                # 1. Diagonal value distribution by category
                fig = px.box(
                    df,
                    x='category',
                    y='diagonal_value',
                    color='category',
                    title=f'OV Circuit Diagonal Values by Token Category (L{layer}H{head})',
                    labels={'diagonal_value': 'Diagonal Value', 'category': 'Token Category'},
                    points='all',
                    hover_data=['token', 'token_id', 'is_negative', 'normalized_rank']
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                plots['diagonal_by_category'] = fig

                # Other plots omitted for brevity...
            except Exception as e:
                print(f"Error creating visualizations: {e}")

        return {
            'layer': layer,
            'head': head,
            'token_data': df,
            'category_metrics': category_metrics,
            'overall_metrics': overall_metrics,
            'plots': plots if visualize else None
        }

    def compare_suppression_across_layers(self, tokens, layers=None, heads=None,
                                        sample_size=1000, visualize=True, threshold=0.5):
        """
        Compare suppression mechanisms across multiple layers/heads to identify patterns
        of suppression behavior throughout the model.

        Args:
            tokens: Input tokens
            layers: List of layers to analyze (if None, use all layers)
            heads: List of heads to analyze (if None, use all heads)
            sample_size: Number of tokens to sample for efficiency
            visualize: Whether to create visualizations
            threshold: Threshold for classifying suppression behavior

        Returns:
            Dictionary with cross-layer analysis results and metrics
        """        
        # Choose layers if not specified
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        # Choose heads if not specified
        if heads is None:
            heads = list(range(self.model.cfg.n_heads))

        # Results storage
        results = []

        # Analyze each layer/head
        for layer in layers:
            for head in heads:
                try:
                    # Run weights-based verification
                    verification_results = self.weights_based_verification(
                        tokens=tokens,
                        layer=layer,
                        head=head,
                        visualize=False,
                        sample_size=sample_size
                    )

                    # Store main results
                    head_results = {
                        "layer": layer,
                        "head": head,
                        "diag_negative_ratio": verification_results["diag_negative_ratio"],
                        "bottom_5_percent_ratio": verification_results["bottom_5_percent_ratio"],
                        "suppression_score": verification_results["suppression_score"],
                        "is_suppression_head": verification_results["is_suppression_head"]
                    }

                    results.append(head_results)

                except Exception as e:
                    print(f"Error analyzing L{layer}H{head}: {e}")
                    # Add placeholder with empty results
                    results.append({
                        "layer": layer,
                        "head": head,
                        "diag_negative_ratio": 0,
                        "bottom_5_percent_ratio": 0,
                        "suppression_score": 0,
                        "is_suppression_head": False
                    })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Create visualizations
        plots = {}
        if visualize and not results_df.empty:
            try:
                # 1. Heatmap of suppression scores across layers and heads
                pivot_df = results_df.pivot(index="layer", columns="head", values="suppression_score")

                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Head", y="Layer", color="Suppression Score"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale="Blues",
                    title="Suppression Score Across Layers and Heads"
                )

                # Highlight suppression heads
                for i, row in results_df.iterrows():
                    if row["is_suppression_head"]:
                        fig.add_shape(
                            type="rect",
                            x0=row["head"]-0.5,
                            x1=row["head"]+0.5,
                            y0=row["layer"]-0.5,
                            y1=row["layer"]+0.5,
                            line=dict(color="red", width=2),
                            fillcolor="rgba(0,0,0,0)"
                        )

                plots["suppression_heatmap"] = fig

                # 2. Per-layer summary
                layer_summary = results_df.groupby("layer").agg({
                    "diag_negative_ratio": "mean",
                    "bottom_5_percent_ratio": "mean",
                    "suppression_score": "mean",
                    "is_suppression_head": lambda x: np.mean(x) * 100  # Convert to percentage
                }).reset_index()

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Bar(
                        x=layer_summary["layer"],
                        y=layer_summary["suppression_score"],
                        name="Avg Suppression Score",
                        marker_color="blue"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=layer_summary["layer"],
                        y=layer_summary["is_suppression_head"],
                        name="% Suppression Heads",
                        mode="lines+markers",
                        marker_color="red",
                        line=dict(width=3)
                    ),
                    secondary_y=True
                )

                fig.update_layout(
                    title="Suppression Behavior Across Layers",
                    xaxis_title="Layer",
                    yaxis_title="Average Suppression Score",
                    yaxis2_title="% Heads with Suppression",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                plots["layer_summary"] = fig

                # 3. Distribution of suppression scores
                fig = px.histogram(
                    results_df,
                    x="suppression_score",
                    color="is_suppression_head",
                    marginal="rug",
                    hover_data=results_df.columns,
                    title="Distribution of Suppression Scores"
                )

                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold})"
                )

                plots["score_distribution"] = fig

            except Exception as e:
                print(f"Error creating visualizations: {e}")
                traceback.print_exc()

        return {
            "results_df": results_df,
            "top_suppression_heads": results_df.sort_values("suppression_score", ascending=False).head(5),
            "suppression_head_count": results_df["is_suppression_head"].sum(),
            "plots": plots if visualize else None
        }

    def compare_models_across_layers(self, model_specs, prompt, layer_ranges=None,
                              sample_size=1000, threshold=0.6, save_visualizations=False,
                              output_dir="model_layer_analysis"):
        """
        Compare suppression mechanisms across layers for multiple models,
        extending the compare_suppression_across_layers function to work with multiple models.

        Args:
            model_specs: Dictionary mapping model names to metadata (or can be just a list of model names)
            prompt: Text prompt to use for analysis
            layer_ranges: Dictionary mapping model names to (start_layer, end_layer) tuples
                        If None, analyzes all layers for each model
            sample_size: Number of tokens to sample for weight matrix analysis
            threshold: Threshold for classifying suppression behavior
            save_visualizations: Whether to save visualizations to files
            output_dir: Directory to save visualizations if saving

        Returns:
            Dictionary with comparative results across models and layers
        """        
        # Process model_specs to standardized format
        model_names = []
        if isinstance(model_specs, dict):
            model_names = list(model_specs.keys())
        elif isinstance(model_specs, (list, tuple)):
            model_names = list(model_specs)
        elif hasattr(model_specs, '__iter__') and not isinstance(model_specs, str):
            # Handle iterable objects like dict_keys
            model_names = list(model_specs)
        else:
            raise ValueError("model_specs must be a dictionary, list, or iterable of model names")

        # Create output directory if saving visualizations
        if save_visualizations and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store results for all models
        all_results = {}
        all_layer_data = []
        original_model = self.model

        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Analyzing model: {model_name}")
            print(f"{'='*50}")

            try:
                # Load model
                model = HookedTransformer.from_pretrained(model_name, device=self.device)

                # Process tokens
                tokens = model.to_tokens(prompt, prepend_bos=True)

                # Create analyzer for this model
                model_analyzer = CircuitAnalyzer(model)

                # Determine layer range for this model
                if layer_ranges and model_name in layer_ranges:
                    start_layer, end_layer = layer_ranges[model_name]
                else:
                    start_layer, end_layer = 0, model.cfg.n_layers - 1

                # Select a representative subset of layers if there are many
                if end_layer - start_layer > 10:
                    # Choose ~10 evenly spaced layers
                    num_samples = min(10, end_layer - start_layer + 1)
                    layers = np.linspace(start_layer, end_layer, num_samples, dtype=int).tolist()
                else:
                    layers = list(range(start_layer, end_layer + 1))

                print(f"Analyzing layers: {layers}")

                # Run comparison for this model
                model_results = model_analyzer.compare_suppression_across_layers(
                    tokens=tokens,
                    layers=layers,
                    heads=None,  # Analyze all heads
                    sample_size=sample_size,
                    visualize=save_visualizations,
                    threshold=threshold
                )

                # Store results for this model
                all_results[model_name] = model_results

                # Add model name to results DataFrame
                if "results_df" in model_results:
                    model_df = model_results["results_df"].copy()
                    model_df["model"] = model_name
                    model_df["total_layers"] = model.cfg.n_layers
                    model_df["normalized_layer"] = model_df["layer"] / model.cfg.n_layers
                    all_layer_data.append(model_df)

                # Save visualizations if requested
                if save_visualizations and "plots" in model_results:
                    for plot_name, plot in model_results["plots"].items():
                        if hasattr(plot, "write_image"):
                            file_path = os.path.join(output_dir,
                                f"{model_name.replace('/', '_')}_{plot_name}.png")
                            plot.write_image(file_path)

                # Clean up
                del model
                del model_analyzer
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error analyzing model {model_name}: {e}")
                traceback.print_exc()
                all_results[model_name] = {"error": str(e)}

        # Combine all results into a master DataFrame
        if all_layer_data:
            combined_df = pd.concat(all_layer_data, ignore_index=True)
            all_results["combined_df"] = combined_df

            # Create comparative visualizations
            if save_visualizations:
                try:
                    # 1. Suppression score across layers by model
                    fig = px.line(
                        combined_df.groupby(["model", "layer"]).agg({
                            "suppression_score": "mean",
                            "normalized_layer": "first"
                        }).reset_index(),
                        x="layer",
                        y="suppression_score",
                        color="model",
                        title="Average Suppression Score Across Layers",
                        labels={"suppression_score": "Suppression Score", "layer": "Layer"},
                        markers=True
                    )

                    file_path = os.path.join(output_dir, "comparative_suppression_by_layer.png")
                    fig.write_image(file_path)
                    all_results["plots"] = all_results.get("plots", {})
                    all_results["plots"]["comparative_suppression"] = fig

                    # 2. Normalized layer comparison
                    fig = px.line(
                        combined_df.groupby(["model", "normalized_layer"]).agg({
                            "suppression_score": "mean",
                            "layer": "first"
                        }).reset_index(),
                        x="normalized_layer",
                        y="suppression_score",
                        color="model",
                        title="Suppression Score by Normalized Layer Position",
                        labels={"suppression_score": "Suppression Score", "normalized_layer": "Normalized Layer Position (0-1)"},
                        markers=True
                    )

                    file_path = os.path.join(output_dir, "comparative_normalized_suppression.png")
                    fig.write_image(file_path)
                    all_results["plots"]["comparative_normalized"] = fig

                    # 3. Distribution of suppression scores by model
                    fig = px.box(
                        combined_df,
                        x="model",
                        y="suppression_score",
                        color="model",
                        title="Distribution of Suppression Scores by Model",
                        points="all"
                    )

                    file_path = os.path.join(output_dir, "suppression_distribution_by_model.png")
                    fig.write_image(file_path)
                    all_results["plots"]["score_distribution"] = fig

                    # 4. Proportion of suppression heads by model
                    suppression_by_model = combined_df.groupby("model").agg({
                        "is_suppression_head": ["mean", "sum"],
                        "head": "count"
                    }).reset_index()

                    suppression_by_model.columns = ["model", "suppression_ratio", "suppression_count", "total_heads"]
                    suppression_by_model["percentage"] = suppression_by_model["suppression_ratio"] * 100

                    fig = px.bar(
                        suppression_by_model,
                        x="model",
                        y="percentage",
                        color="model",
                        title="Percentage of Suppression Heads by Model",
                        text_auto=".1f",
                        labels={"percentage": "Percentage", "model": "Model"}
                    )

                    file_path = os.path.join(output_dir, "suppression_percentage_by_model.png")
                    fig.write_image(file_path)
                    all_results["plots"]["suppression_percentage"] = fig

                except Exception as e:
                    print(f"Error creating comparative visualizations: {e}")                    
                    traceback.print_exc()

        # Restore original model
        self.model = original_model

        return all_results

    def analyze_all_components_across_models(self, model_specs, prompt, specific_heads=None,
                                          run_all_analyses=True, save_results=True,
                                          output_dir="comprehensive_analysis"):
        """
        Perform a comprehensive analysis of suppression mechanisms across multiple models,
        running all available analysis methods on specified heads.

        Args:
            model_specs: Dictionary mapping model names to metadata (can include specified heads)
                        If 'heads' not in metadata, uses specific_heads or analyzes random samples
            prompt: Text prompt to use for analysis
            specific_heads: Default heads to analyze for models that don't specify them
                          Format: [(layer, head, description), ...]
            run_all_analyses: Whether to run all available analyses (if False, runs only basic analysis)
            save_results: Whether to save results and visualizations
            output_dir: Directory to save outputs

        Returns:
            Dictionary with comprehensive analysis results
        """        
        # Create output directory
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process model_specs to ensure consistent format
        processed_specs = {}
        for model_name, model_meta in model_specs.items():
            if isinstance(model_meta, list):
                # If model_meta is a list, assume it's a list of (layer, head, desc) tuples
                processed_specs[model_name] = {"heads": model_meta}
            elif isinstance(model_meta, dict):
                processed_specs[model_name] = model_meta
            else:
                processed_specs[model_name] = {}

        model_specs = processed_specs

        # Initialize results structure
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "comparative": {}
        }

        original_model = self.model

        # Process each model
        for model_name, model_meta in model_specs.items():
            print(f"\n{'='*50}")
            print(f"Analyzing model: {model_name}")
            print(f"{'='*50}")

            try:
                # Load model
                model = HookedTransformer.from_pretrained(model_name, device=self.device)

                # Process tokens
                tokens = model.to_tokens(prompt, prepend_bos=True)

                # Create analyzer for this model
                model_analyzer = CircuitAnalyzer(model)

                # Determine heads to analyze
                heads_to_analyze = []

                if "heads" in model_meta and model_meta["heads"]:
                    # Use specified heads
                    heads_to_analyze = model_meta["heads"]
                elif specific_heads:
                    # Use default specific heads
                    heads_to_analyze = specific_heads
                else:
                    # Sample random heads
                    n_layers = model.cfg.n_layers
                    n_heads = model.cfg.n_heads

                    # Sample from early, middle, and late layers
                    early_layers = list(range(0, n_layers // 3))
                    mid_layers = list(range(n_layers // 3, 2 * n_layers // 3))
                    late_layers = list(range(2 * n_layers // 3, n_layers))

                    heads_to_analyze = []
                    for layer_group in [early_layers, mid_layers, late_layers]:
                        if layer_group:
                            layer = random.choice(layer_group)
                            head = random.randint(0, n_heads - 1)
                            heads_to_analyze.append((layer, head, f"Random L{layer}H{head}"))

                print(f"Analyzing heads: {heads_to_analyze}")

                # Initialize results for this model
                model_results = {
                    "model_info": {
                        "name": model_name,
                        "n_layers": model.cfg.n_layers,
                        "n_heads": model.cfg.n_heads,
                        "d_model": model.cfg.d_model
                    },
                    "heads": {}
                }

                # Analyze each head
                for layer, head, description in heads_to_analyze:
                    print(f"\nAnalyzing {model_name} - L{layer}H{head} ({description})")

                    # Initialize results for this head
                    head_results = {
                        "layer": layer,
                        "head": head,
                        "description": description,
                        "analyses": {}
                    }

                    # 1. Weights-based verification
                    try:
                        print("Running weights-based verification...")
                        verification_results = model_analyzer.weights_based_verification(
                            tokens=tokens,
                            layer=layer,
                            head=head,
                            visualize=save_results,
                            sample_size=2000
                        )

                        # Extract key metrics
                        head_results["analyses"]["weights_verification"] = {
                            key: verification_results[key]
                            for key in ["diag_negative_ratio", "bottom_5_percent_ratio",
                                      "top_10_negative_ratio", "median_rank",
                                      "suppression_score", "is_suppression_head"]
                            if key in verification_results
                        }

                        # Save visualizations
                        if save_results and "plots" in verification_results:
                            for plot_name, plot in verification_results["plots"].items():
                                if hasattr(plot, "write_image"):
                                    file_path = os.path.join(output_dir,
                                        f"{model_name.replace('/', '_')}_L{layer}H{head}_weights_{plot_name}.png")
                                    plot.write_image(file_path)

                        print(f"  Suppression score: {verification_results.get('suppression_score', 0):.4f}")
                        print(f"  Is suppression head: {verification_results.get('is_suppression_head', False)}")

                    except Exception as e:
                        print(f"Error in weights-based verification: {e}")
                        head_results["analyses"]["weights_verification"] = {"error": str(e)}

                    # Run additional analyses if requested
                    if run_all_analyses:
                        # 2. Perpendicular component analysis
                        try:
                            print("Running perpendicular component analysis...")
                            perp_results = model_analyzer.perpendicular_component_analysis(
                                tokens=tokens,
                                layer=layer,
                                head=head,
                                visualize=save_results
                            )

                            # Extract key metrics
                            if "combined_metrics" in perp_results:
                                head_results["analyses"]["perpendicular_component"] = {
                                    "combined_metrics": perp_results["combined_metrics"]
                                }

                                print(f"  Perpendicular dominance: {perp_results['combined_metrics'].get('perpendicular_dominance', False)}")

                            # Save visualizations
                            if save_results and "plots" in perp_results:
                                for plot_name, plot in perp_results["plots"].items():
                                    if hasattr(plot, "write_image"):
                                        file_path = os.path.join(output_dir,
                                            f"{model_name.replace('/', '_')}_L{layer}H{head}_perp_{plot_name}.png")
                                        plot.write_image(file_path)

                        except Exception as e:
                            print(f"Error in perpendicular component analysis: {e}")
                            head_results["analyses"]["perpendicular_component"] = {"error": str(e)}

                        # 3. OV Negative Diagonal Patterns
                        try:
                            print("Analyzing OV negative diagonal patterns...")
                            diagonal_results = model_analyzer.analyze_ov_negative_diagonal_patterns(
                                tokens=tokens,
                                layer=layer,
                                head=head,
                                visualize=save_results
                            )

                            # Extract key metrics
                            if "overall_metrics" in diagonal_results:
                                head_results["analyses"]["ov_negative_diagonal"] = {
                                    "overall_metrics": diagonal_results["overall_metrics"]
                                }

                                # Add category summary
                                if "category_metrics" in diagonal_results:
                                    category_summary = {}
                                    for _, row in diagonal_results["category_metrics"].iterrows():
                                        category = row["category"]
                                        category_summary[category] = {
                                            "count": int(row["count"]),
                                            "negative_ratio": float(row["negative_ratio"]),
                                            "bottom_5_pct_ratio": float(row["bottom_5_pct_ratio"])
                                        }
                                    head_results["analyses"]["ov_negative_diagonal"]["category_summary"] = category_summary

                                print(f"  Overall negative diagonal ratio: {diagonal_results['overall_metrics']['diag_negative_ratio']:.4f}")

                            # Save visualizations
                            if save_results and "plots" in diagonal_results:
                                for plot_name, plot in diagonal_results["plots"].items():
                                    if hasattr(plot, "write_image"):
                                        file_path = os.path.join(output_dir,
                                            f"{model_name.replace('/', '_')}_L{layer}H{head}_diagonal_{plot_name}.png")
                                        plot.write_image(file_path)

                        except Exception as e:
                            print(f"Error in OV negative diagonal analysis: {e}")
                            head_results["analyses"]["ov_negative_diagonal"] = {"error": str(e)}

                    # Add results for this head
                    model_results["heads"][f"L{layer}H{head}"] = head_results

                # Add model to results
                all_results["models"][model_name] = model_results

                # Clean up
                del model
                del model_analyzer
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error analyzing model {model_name}: {e}")
                all_results["models"][model_name] = {"error": str(e)}

        # Run comparative layer analysis if there are multiple models
        if len(model_specs) > 1 and run_all_analyses:
            try:
                print("\nRunning comparative layer analysis across models...")

                # Simplified model specs for layer analysis
                layer_model_specs = list(model_specs.keys())

                # Run comparison
                layer_comparison = self.compare_models_across_layers(
                    model_specs=layer_model_specs,
                    prompt=prompt,
                    save_visualizations=save_results,
                    output_dir=output_dir
                )

                # Add to results
                all_results["comparative"]["layer_analysis"] = {
                    "summary": {
                        model_name: {
                            "suppression_head_count": results.get("suppression_head_count", 0)
                        } for model_name, results in layer_comparison.items()
                        if isinstance(results, dict) and "suppression_head_count" in results
                    }
                }

            except Exception as e:
                print(f"Error in comparative layer analysis: {e}")
                all_results["comparative"]["layer_analysis"] = {"error": str(e)}

        # Create summary dataframe
        summary_rows = []
        for model_name, model_data in all_results["models"].items():
            if "heads" in model_data:
                for head_key, head_data in model_data["heads"].items():
                    row = {
                        "model": model_name,
                        "head": head_key,
                        "description": head_data.get("description", "")
                    }

                    # Add weights verification metrics
                    if "analyses" in head_data and "weights_verification" in head_data["analyses"]:
                        for metric, value in head_data["analyses"]["weights_verification"].items():
                            if metric != "error":
                                row[f"weights_{metric}"] = value

                    # Add perpendicular component metrics
                    if "analyses" in head_data and "perpendicular_component" in head_data["analyses"]:
                        if "combined_metrics" in head_data["analyses"]["perpendicular_component"]:
                            for metric, value in head_data["analyses"]["perpendicular_component"]["combined_metrics"].items():
                                row[f"perp_{metric}"] = value

                    # Add negative diagonal metrics
                    if "analyses" in head_data and "ov_negative_diagonal" in head_data["analyses"]:
                        if "overall_metrics" in head_data["analyses"]["ov_negative_diagonal"]:
                            for metric, value in head_data["analyses"]["ov_negative_diagonal"]["overall_metrics"].items():
                                row[f"diag_{metric}"] = value

                    summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        all_results["summary_df"] = summary_df.to_dict()

        # Save results to file
        if save_results:
            try:
                # Create a serializable version
                serializable_results = {
                    "timestamp": all_results["timestamp"],
                    "models": {},
                    "comparative": all_results["comparative"]
                }

                # Process each model
                for model_name, model_data in all_results["models"].items():
                    serializable_results["models"][model_name] = {
                        "model_info": model_data.get("model_info", {}),
                        "heads": {}
                    }

                    if "heads" in model_data:
                        for head_key, head_data in model_data["heads"].items():
                            serializable_results["models"][model_name]["heads"][head_key] = head_data

                # Save to file
                results_file = os.path.join(output_dir, "comprehensive_results.json")
                with open(results_file, "w") as f:
                    json.dump(serializable_results, f, indent=2)

                # Also save the summary dataframe
                if not summary_df.empty:
                    summary_file = os.path.join(output_dir, "summary_results.csv")
                    summary_df.to_csv(summary_file, index=False)

                print(f"\nResults saved to {output_dir}")

            except Exception as e:
                print(f"Error saving results: {e}")

        # Restore original model
        self.model = original_model

        return all_results

    def compute_token_jacobian(model, prompt, focus_token):
        """
        Compute the Jacobian of the focus token logit (at the final position)
        with respect to the input embeddings.
        This measures which input tokens most influence the focus token prediction.
        """
        # Convert prompt to tokens (with BOS prepended)
        tokens = model.to_tokens(prompt, prepend_bos=True)
        # Determine focus token id (assumes model.tokenizer is available)
        if hasattr(model, 'tokenizer'):
            focus_token_id = model.tokenizer.encode(focus_token)[0]
        else:
            focus_token_id = int(focus_token)

        # Use the embedding hook to capture the input embeddings
        emb_activations = None
        def hook(module, inp, out):
            nonlocal emb_activations
            emb_activations = out
        hook_handle = model.hook_embed.register_forward_hook(hook)
        _ = model(tokens)
        hook_handle.remove()

        # Make the captured embeddings differentiable
        emb = emb_activations.clone().detach().requires_grad_(True)

        # Forward-pass manually from embeddings through the transformer blocks
        x = emb
        for block in model.blocks:
            x = block(x)
        x = model.final_ln(x)
        logits = x @ model.W_U  # unembedding
        logit_focus = logits[0, -1, focus_token_id]

        # Compute Jacobian: derivative of focus token logit with respect to each input token embedding
        jacobian = torch.autograd.grad(logit_focus, emb, retain_graph=False)[0]
        # Return the sensitivity per input position (as a numpy array)
        return jacobian[0].cpu().numpy()

    def compute_qk_attention(model, prompt, layer_idx, head_idx):
        """
        Compute the QK attention pattern for a given layer and head.
        """
        tokens = model.to_tokens(prompt, prepend_bos=True)
        # Retrieve the cached attention pattern using a filter on hook names
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: f"blocks.{layer_idx}.attn.hook_pattern" in name)
        pattern_key = f"blocks.{layer_idx}.attn.hook_pattern"
        if pattern_key not in cache:
            pattern_key = f"transformer.h.{layer_idx}.attn.hook_pattern"
        # Return the attention pattern for the specified head (shape: [seq, seq])
        return cache[pattern_key][0, head_idx].detach().cpu().numpy()

    def compute_ov_eigendecomposition(model, layer_idx, head_idx):
        """
        Compute the OV circuit matrix for the specified head and return its eigenvalue decomposition.
        The OV matrix is computed as W_O @ W_U.
        Since W_O is [d_head, d_model] and W_U is [d_model, vocab_size], we form a square
        matrix by computing OV @ OV^T.
        """
        # Extract the output weight matrix W_O using the appropriate attribute
        if hasattr(model, 'blocks'):
            W_O = model.blocks[layer_idx].attn.W_O[head_idx].detach()
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            W_O = model.transformer.h[layer_idx].attn.c_proj.weight.view(
                model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
            )[head_idx].detach()
        else:
            raise ValueError("Model architecture not supported for OV eigendecomposition.")
        # Get unembedding matrix W_U
        W_U = model.W_U.detach()
        # Compute the OV circuit matrix (projected unembedding)
        OV = W_O @ W_U  # [d_head, vocab_size]
        # Form a square matrix by OV @ OV^T and compute its eigenvalues
        ov_square = OV @ OV.T  # [d_head, d_head]
        eigenvalues, _ = np.linalg.eig(ov_square.cpu().numpy())
        return OV, eigenvalues

    def visualize_ov_transformation_on_axes(model, layer_idx, head_idx, tokens, ax):
        """
        Visualize the OV transformation for a given set of tokens in 3D.
        The function projects the token embeddings (via W_O) into a low-dimensional space using PCA.
        """
        # Convert tokens to token ids
        if hasattr(model, 'tokenizer'):
            token_ids = [model.tokenizer.encode(t)[0] for t in tokens]
        else:
            token_ids = [int(t) for t in tokens]
        token_ids = torch.tensor(token_ids, device=model.W_E.device)

        # Get effective token embeddings from the embedding matrix
        effective_WE = model.W_E[token_ids]  # [n_tokens, d_model]

        # Extract W_O for the specified head
        if hasattr(model, 'blocks'):
            W_O = model.blocks[layer_idx].attn.W_O[head_idx].detach()
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            W_O = model.transformer.h[layer_idx].attn.c_proj.weight.view(
                model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
            )[head_idx].detach()
        else:
            raise ValueError("Model architecture not supported for OV transformation visualization.")

        # Compute the OV transformation for the tokens
        ov_transformed = effective_WE @ W_O.T  # [n_tokens, d_head]
        ov_transformed_np = ov_transformed.cpu().numpy()

        # Reduce the dimensions to 3 using PCA for 3D visualization
        pca = PCA(n_components=3)
        ov_3d = pca.fit_transform(ov_transformed_np)

        # Plot the 3D scatter and annotate with token strings
        ax.scatter(ov_3d[:, 0], ov_3d[:, 1], ov_3d[:, 2])
        for i, t in enumerate(tokens):
            ax.text(ov_3d[i, 0], ov_3d[i, 1], ov_3d[i, 2], t)
        ax.set_title("3D OV Transformation (PCA-reduced)")

    def compute_jacobian_eigenvalue_correlation(jacobian, eigenvalues):
        """
        Compute the Pearson correlation between the binned sensitivity of the Jacobian
        (projected into a one-dimensional summary per input token) and the magnitude of eigenvalues.
        The Jacobian is binned (by averaging over tokens) to match the number of eigenvalues.
        """
        # Compute L2 norm (sensitivity) for each input token embedding
        jacobian_norm = np.linalg.norm(jacobian, axis=1)  # [seq_len]
        num_bins = len(eigenvalues)
        # If there are more tokens than eigenvalues, bin the sensitivity values evenly
        if len(jacobian_norm) >= num_bins:
            binned = np.array([np.mean(bin) for bin in np.array_split(jacobian_norm, num_bins)])
        else:
            binned = jacobian_norm
        eigen_mag = np.abs(eigenvalues)
        # Ensure both vectors are of equal length by truncation
        min_len = min(len(binned), len(eigen_mag))
        binned = binned[:min_len]
        eigen_mag = eigen_mag[:min_len]
        corr, _ = pearsonr(binned, eigen_mag)
        return corr

    def visualize_jacobian_ov_relationship(model, prompt, layer_idx, head_idx, focus_token):
        """
        Visualize the relationship between Jacobian sensitivity and the OV eigenspace.
        Creates a 2x2 subplot figure:
          - Top left: Jacobian sensitivity heatmap for the focus token.
          - Top right: QK attention heatmap for the specified layer and head.
          - Bottom left: OV eigenvalue spectrum with bars colored by sign.
          - Bottom right: 3D OV transformation visualization (PCA-reduced) for selected tokens.
        Also computes and displays the Pearson correlation between the Jacobian and eigenvalue patterns.
        """
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Top left: Compute and visualize Jacobian with respect to inputs
        ax1 = fig.add_subplot(2, 2, 1)
        jacobian = compute_token_jacobian(model, prompt, focus_token)
        im = ax1.imshow(jacobian, cmap='RdBu_r', interpolation='none')
        ax1.set_title(f'Jacobian Sensitivity for Token: {focus_token}')
        plt.colorbar(im, ax=ax1)

        # 2. Top right: QK attention heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        qk_attention = compute_qk_attention(model, prompt, layer_idx, head_idx)
        im = ax2.imshow(qk_attention, cmap='viridis', interpolation='none')
        ax2.set_title(f'QK Attention Pattern (Layer {layer_idx}, Head {head_idx})')
        plt.colorbar(im, ax=ax2)

        # 3. Bottom left: OV eigenvalue spectrum
        ax3 = fig.add_subplot(2, 2, 3)
        ov_matrix, eigenvalues = compute_ov_eigendecomposition(model, layer_idx, head_idx)
        bars = ax3.bar(range(len(eigenvalues)), eigenvalues.real)
        # Color bars by sign
        for i, bar in enumerate(bars):
            bar.set_color('red' if eigenvalues[i].real < 0 else 'green')
        ax3.set_title(f'OV Eigenvalue Spectrum (Layer {layer_idx}, Head {head_idx})')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 4. Bottom right: 3D OV transformation visualization
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        visualize_ov_transformation_on_axes(model, layer_idx, head_idx, tokens=[focus_token, "basket", "box"], ax=ax4)

        # Compute and display the correlation between Jacobian sensitivity and eigenvalue pattern
        correlation = compute_jacobian_eigenvalue_correlation(jacobian, eigenvalues)
        fig.text(0.5, 0.01, f'Correlation between Jacobian sensitivity and eigenvalue pattern: {correlation:.4f}',
                ha='center', fontsize=14)

        return fig
