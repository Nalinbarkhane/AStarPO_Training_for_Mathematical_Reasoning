import os
import torch
from tqdm import tqdm

from pag_rl import (
    Sampler, RewardFunction, MetricsTracker,
    load_jsonl, set_seed, load_config
)

def get_question_and_answer(item):
    """
    Extract question and answer from data item.
    Handles different data formats.
    """
    question = None
    answer = None
    
    if 'problem' in item:
        question = item['problem']
    elif 'question' in item:
        question = item['question']
    elif 'prompt' in item:
        question = item['prompt']
    
    if 'answer' in item:
        answer = item['answer']
    elif 'solution' in item:
        answer = item['solution']
    elif 'ground_truth' in item:
        answer = item['ground_truth']
    
    if question is None or answer is None:
        raise ValueError(f"Cannot find question/answer in item. Available fields: {item.keys()}")
    
    return question, answer

def evaluate():
    """Main evaluation function."""
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Set seed
    set_seed(42)
    
    print("="*50)
    print("Starting Evaluation")
    print("="*50)
    
    # Determine which model to evaluate
    checkpoint_dir = os.path.join(config['output_dir'], f"epoch_{config['num_epochs']}")
    
    if os.path.exists(checkpoint_dir):
        model_path = checkpoint_dir
        print(f"\nEvaluating TRAINED model from: {checkpoint_dir}")
    else:
        model_path = config['model_name']
        print(f"\nEvaluating BASE model: {model_path}")
    
    # Initialize sampler
    print("Loading model...")
    sampler = Sampler(
        model_name=model_path,
        device=config['device']
    )
    
    # Initialize reward function
    reward_fn = RewardFunction()
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_jsonl(config['eval_data'])
    print(f"Loaded {len(eval_data)} evaluation examples")
    
    # Check data format
    sample_question, sample_answer = get_question_and_answer(eval_data[0])
    print(f"✓ Data format validated")
    
    # Evaluation loop
    print("\n" + "="*50)
    print("Running Evaluation")
    print("="*50 + "\n")
    
    sampler.model.eval()
    metrics_tracker = MetricsTracker()
    
    results = []
    
    with torch.no_grad():
        for idx, item in enumerate(tqdm(eval_data, desc="Evaluating")):
            question, ground_truth = get_question_and_answer(item)
            
            # Generate solution
            solution = sampler.generate_single(
                question,
                max_new_tokens=config['max_new_tokens'],
                temperature=0.1,  # Lower temperature for evaluation
                top_p=0.95
            )
            
            # Verify correctness
            is_correct = reward_fn.verifier.verify(solution, ground_truth)
            
            # Update metrics
            metrics_tracker.update(
                num_correct=1 if is_correct else 0,
                num_samples=1
            )
            
            # Store result
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'generated_solution': solution,
                'correct': is_correct
            })
            
            # Print first 3 examples
            if idx < 3:
                print(f"\n{'='*50}")
                print(f"Example {idx + 1}:")
                print(f"{'='*50}")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Generated: {solution[:300]}...")
                print(f"Correct: {'✓' if is_correct else '✗'}")
    
    # Final summary
    summary = metrics_tracker.summary()
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Accuracy: {summary['accuracy']:.2%} ({summary['total_correct']}/{summary['total_samples']})")
    print("="*70)
    
    # Save results
    results_path = os.path.join(config['log_dir'], 'eval_results.jsonl')
    from pag_rl import save_jsonl
    save_jsonl(results, results_path)
    print(f"\nDetailed results saved to: {results_path}")
    
    return summary['accuracy']

if __name__ == "__main__":
    evaluate()