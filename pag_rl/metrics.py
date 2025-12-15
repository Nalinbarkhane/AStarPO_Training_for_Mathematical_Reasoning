class MetricsTracker:
    def __init__(self):
        """
        Track training and evaluation metrics.
        """
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_correct = 0
        self.total_samples = 0
        self.rewards = []
        self.losses = []
    
    def update(self, num_correct, num_samples, reward=None, loss=None):
        """
        Update metrics.
        
        Args:
            num_correct: Number of correct predictions
            num_samples: Total number of samples
            reward: Reward value (optional)
            loss: Loss value (optional)
        """
        self.total_correct += num_correct
        self.total_samples += num_samples
        
        if reward is not None:
            self.rewards.append(reward)
        
        if loss is not None:
            self.losses.append(loss)
    
    def get_accuracy(self):
        """Get current accuracy."""
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples
    
    def get_mean_reward(self):
        """Get mean reward."""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    def get_mean_loss(self):
        """Get mean loss."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)
    
    def summary(self):
        """Get summary of all metrics."""
        return {
            'accuracy': self.get_accuracy(),
            'mean_reward': self.get_mean_reward(),
            'mean_loss': self.get_mean_loss(),
            'total_correct': self.total_correct,
            'total_samples': self.total_samples
        }