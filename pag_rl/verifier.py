import re

class Verifier:
    def __init__(self):
        """
        Initialize the verifier for checking math solutions.
        """
        pass
    
    def extract_answer(self, solution):
        """
        Extract the final answer from a solution.
        
        Looks for common patterns:
        - \\boxed{answer}
        - Final answer: answer
        - The answer is answer
        
        Args:
            solution: The generated solution text
            
        Returns:
            Extracted answer string or None
        """
        if not solution:
            return None
        
        # Pattern 1: \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, solution)
        if matches:
            return matches[-1].strip()  # Return last boxed answer
        
        # Pattern 2: Final answer is: ...
        final_patterns = [
            r'[Ff]inal [Aa]nswer[:\s]+([^\n\.]+)',
            r'[Ff]inal [Aa]nswer[:\s]*\\boxed\{([^}]+)\}',
        ]
        for pattern in final_patterns:
            match = re.search(pattern, solution)
            if match:
                return match.group(1).strip()
        
        # Pattern 3: The answer is ...
        answer_patterns = [
            r'[Tt]he answer is[:\s]+([^\n\.]+)',
            r'[Aa]nswer[:\s]+([^\n\.]+)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, solution)
            if match:
                return match.group(1).strip()
        
        # Pattern 4: Therefore, ... (last sentence)
        therefore_pattern = r'[Tt]herefore[,:\s]+([^\n\.]+)'
        match = re.search(therefore_pattern, solution)
        if match:
            return match.group(1).strip()
        
        return None
    
    def normalize_answer(self, answer):
        """
        Normalize an answer for comparison.
        
        Args:
            answer: Answer string
            
        Returns:
            Normalized answer string
        """
        if answer is None:
            return ""
        
        # Convert to string if not already
        answer = str(answer)
        
        # Remove extra whitespace
        answer = answer.strip()
        
        # Remove dollar signs
        answer = answer.replace('$', '')
        
        # Remove commas from numbers
        answer = answer.replace(',', '')
        
        # Remove backslashes
        answer = answer.replace('\\', '')
        
        # Handle fractions: convert \frac{a}{b} to a/b
        frac_pattern = r'frac\{([^}]+)\}\{([^}]+)\}'
        answer = re.sub(frac_pattern, r'\1/\2', answer)
        
        # Remove extra spaces
        answer = ' '.join(answer.split())
        
        # Lowercase for comparison
        answer = answer.lower()
        
        return answer
    
    def answers_equivalent(self, pred, truth):
        """
        Check if two answers are equivalent.
        
        Args:
            pred: Predicted answer
            truth: Ground truth answer
            
        Returns:
            Boolean indicating equivalence
        """
        pred_norm = self.normalize_answer(pred)
        truth_norm = self.normalize_answer(truth)
        
        if not pred_norm or not truth_norm:
            return False
        
        # Direct string match
        if pred_norm == truth_norm:
            return True
        
        # Try to evaluate as numbers
        try:
            pred_val = float(eval(pred_norm))
            truth_val = float(eval(truth_norm))
            return abs(pred_val - truth_val) < 1e-6
        except:
            pass
        
        return False
    
    def verify(self, solution, ground_truth):
        """
        Verify if a solution is correct.
        
        Args:
            solution: Generated solution text
            ground_truth: Correct answer
            
        Returns:
            Boolean indicating correctness
        """
        predicted = self.extract_answer(solution)
        
        if predicted is None:
            return False
        
        return self.answers_equivalent(predicted, ground_truth)