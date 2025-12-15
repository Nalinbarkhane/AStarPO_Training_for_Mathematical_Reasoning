import torch
import torch.nn.functional as F

class AStarPO:
    def __init__(self, model, ref_model, tokenizer, beta=0.1):
        """
        A*-PO: Advantage-based Policy Optimization (similar to A*-PO paper).
        
        Args:
            model: The model being trained (policy)
            ref_model: Reference model (frozen)
            tokenizer: Tokenizer
            beta: KL divergence penalty coefficient
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.ref_model.eval()
    
    def compute_sequence_log_probs(self, model, input_ids, attention_mask):
        """
        Compute log probabilities of sequences under the model.
        
        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities for each sequence [batch_size]
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs of actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Apply mask
        token_log_probs = token_log_probs * shift_mask
        
        # Sum over sequence length
        sequence_log_probs = token_log_probs.sum(dim=1)  # [batch_size]
        
        return sequence_log_probs
    
    def compute_loss(self, input_ids, attention_mask, rewards):
        """
        Compute A*-PO loss.
        
        Loss = -E[advantage * log π(a|s)] + β * KL(π || π_ref)
        
        Args:
            input_ids: Tokenized sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            rewards: Rewards for each sequence [batch_size]
            
        Returns:
            loss: Total loss (scalar)
            metrics: Dictionary of metrics for logging
        """
        # Ensure rewards are on the correct device
        rewards = rewards.to(input_ids.device)
        
        # Compute log probs from current policy
        policy_log_probs = self.compute_sequence_log_probs(
            self.model, input_ids, attention_mask
        )
        
        # Compute log probs from reference policy
        with torch.no_grad():
            ref_log_probs = self.compute_sequence_log_probs(
                self.ref_model, input_ids, attention_mask
            )
        
        # Compute advantages (centered rewards)
        advantages = rewards - rewards.mean()
        
        # Policy gradient loss: maximize advantage-weighted log probs
        pg_loss = -(advantages * policy_log_probs).mean()
        
        # KL divergence penalty
        kl_div = (policy_log_probs - ref_log_probs).mean()
        
        # Total loss
        loss = pg_loss + self.beta * kl_div
        
        # Metrics for logging
        metrics = {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'kl_div': kl_div.item(),
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
        }
        
        return loss, metrics