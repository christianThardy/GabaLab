import gc
import torch


class LLCEstimator:
    """
    Implementation of Local Learning Coefficient (LLC) and refined LLC metrics
    for transformer models.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.sgld_steps = 50
        self.sgld_lr = 0.001
        self.sgld_noise = 0.01
        self.beta = 0.1
        self.gamma = 100.0
        self.n_samples = 5
        
    def compute_loss(self, tokens):
        with torch.no_grad():
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]
                
            logits = self.model(tokens)
            targets = tokens[:, 1:]
            logits = logits[:, :-1]
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            return loss.item()
        
    def run_sgld(self, tokens, param_tensor, n_steps, lr, noise_scale):
        original_param = param_tensor.clone().detach()
        param = param_tensor.clone().detach().requires_grad_(True)
        samples = []
        losses = []
        
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]
            
        sample_interval = 4
        
        for step in range(n_steps):
            if param.grad is not None:
                param.grad.zero_()
                
            logits = self.model(tokens)
            targets = tokens[:, 1:]
            logits = logits[:, :-1]
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            
            with torch.no_grad():
                noise = torch.randn_like(param) * noise_scale * np.sqrt(lr)
                param.data -= lr * param.grad + noise
                param.grad.zero_()
                
                if step > n_steps // 2 and step % sample_interval == 0:
                    samples.append(param.clone().detach())
                    losses.append(loss.item())
                    
            if step % 10 == 0:
                torch.cuda.empty_cache()
                
        param_tensor.data.copy_(original_param)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return samples, losses
        
    def compute_head_llc(self, tokens, layer, head):
        try:
            if hasattr(self.model, 'blocks'):
                W_Q = self.model.blocks[layer].attn.W_Q[head]
                n_kv_heads = getattr(self.model.cfg, 'n_kv_heads', self.model.cfg.n_heads)
                kv_head = min(head * n_kv_heads // self.model.cfg.n_heads, n_kv_heads - 1)
                if hasattr(self.model.blocks[layer].attn, 'W_K'):
                    W_K = self.model.blocks[layer].attn.W_K[kv_head]
                    W_V = self.model.blocks[layer].attn.W_V[kv_head]
                elif hasattr(self.model.blocks[layer].attn, 'W_KV'):
                    W_KV = self.model.blocks[layer].attn.W_KV[kv_head]
                    d_head = self.model.cfg.d_head
                    W_K = W_KV[:, :d_head]
                    W_V = W_KV[:, d_head:]
                else:
                    W_K = None
                    W_V = None
                W_O = self.model.blocks[layer].attn.W_O[head]
                head_params = [W_Q]
                if W_K is not None:
                    head_params.append(W_K)
                if W_V is not None:
                    head_params.append(W_V)
                head_params.append(W_O)
                try:
                    param_tensor = W_Q
                except:
                    param_tensor = W_Q
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                W_O = self.model.transformer.h[layer].attn.c_proj.weight.view(
                    self.model.cfg.n_heads, self.model.cfg.d_head, self.model.cfg.d_model
                )[head]
                param_tensor = W_O
            else:
                raise ValueError(f"Unsupported model architecture for LLC computation")
        except Exception as e:
            print(f"Error accessing head parameters for LLC: {e}")
            return -1.0
            
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]
            
        baseline_loss = self.compute_loss(tokens)
        samples, losses = self.run_sgld(tokens, param_tensor, self.sgld_steps, self.sgld_lr, self.sgld_noise)
        
        if losses:
            expected_loss = sum(losses) / len(losses)
        else:
            expected_loss = baseline_loss
            
        n = tokens.shape[1]
        llc = n * self.beta * (expected_loss - baseline_loss)
        llc = abs(llc)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return llc
        
    def estimate_wrllc(self, tokens, layer, head):
        return self.compute_head_llc(tokens, layer, head)
        
    def estimate_wdrllc(self, tokens1, tokens2, layer, head):
        llc1 = self.compute_head_llc(tokens1, layer, head)
        try:
            if hasattr(self.model, 'blocks'):
                W_Q1 = self.model.blocks[layer].attn.W_Q[head].clone()
                W_O1 = self.model.blocks[layer].attn.W_O[head].clone()
                
                if tokens2.shape[1] > 512:
                    tokens2 = tokens2[:, :512]
                    
                loss1_on_dist2 = self.compute_loss(tokens2)
                param_tensor = W_Q1
                samples, losses = self.run_sgld(tokens2, param_tensor, self.sgld_steps, self.sgld_lr, self.sgld_noise)
                
                if losses:
                    expected_loss = sum(losses) / len(losses)
                else:
                    expected_loss = loss1_on_dist2
                    
                n = tokens2.shape[1]
                wdrllc = n * self.beta * (expected_loss - loss1_on_dist2)
                wdrllc = abs(wdrllc)
                
                torch.cuda.empty_cache()
                gc.collect()
                
                return wdrllc
            else:
                llc2 = self.compute_head_llc(tokens2, layer, head)
                return abs(llc1 - llc2)
        except Exception as e:
            print(f"Error computing wdrLLC: {e}")
            return abs(llc1)
        
    def compute_circuit_rank(self, components):
        if "W_Q" not in components or "W_O" not in components or components["W_Q"] is None or components["W_O"] is None:
            return -1
        try:
            with torch.no_grad():
                W_Q = components["W_Q"].detach()
                W_O = components["W_O"].detach()
                if W_Q.ndim == 2 and W_O.ndim == 2 and W_Q.shape[1] == W_O.shape[0]:
                    QO = torch.matmul(W_Q, W_O)
                    QO = QO.to(torch.float32)
                    if QO.numel() > 10000:
                        S = torch.linalg.svdvals(QO)
                    else:
                        U, S, V = torch.svd(QO)
                    threshold = 1e-5
                    rank = torch.sum(S > threshold).item()
                    return rank
                else:
                    print(f"Incompatible dimensions: W_Q={W_Q.shape}, W_O={W_O.shape}")
                    return -1
        except Exception as e:
            print(f"Error computing circuit rank: {e}")
            return -1
