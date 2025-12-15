import torch
from .verifier import Verifier

class RewardFunction:
    def __init__(self, correct_reward=1.0, incorrect_reward=-1.0):
        """
        Initialize reward function.
        
        Args:
            correct_reward: Reward for correct answers
            incorrect_reward: Penalty for incorrect answers
        """
        self.verifier = Verifier()
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
    
    def calculate_rewards(self, solutions, ground_truth):
        """
        Calculate rewards for a batch of solutions.
        
        Args:
            solutions: List of generated solutions
            ground_truth: The correct answer
            
        Returns:
            Tensor of rewards
        """
        rewards = []
        
        for solution in solutions:
            is_correct = self.verifier.verify(solution, ground_truth)
            reward = self.correct_reward if is_correct else self.incorrect_reward
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def get_accuracy(self, solutions, ground_truth):
        """
        Calculate accuracy for a batch of solutions.
        
        Args:
            solutions: List of generated solutions
            ground_truth: The correct answer
            
        Returns:
            Accuracy (float between 0 and 1)
        """
        correct = sum(
            1 for sol in solutions 
            if self.verifier.verify(sol, ground_truth)
        )
        return correct / len(solutions) if solutions else 0.0