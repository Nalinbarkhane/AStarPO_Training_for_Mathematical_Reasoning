import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import copy

from pag_rl import (
    Sampler, RewardFunction, AStarPO, MetricsTracker,
    load_jsonl, set_seed, load_config
)

def get_question_and_answer(item):
    """
    Extract question and answer from data item.
    Handles different data formats.
    """
    # Try different field name combinations
    question = None
    answer = None
    
    # Format 1: problem + answer
    if 'problem' in item:
        question = item['problem']
    elif 'question' in item:
        question = item['question']
    elif 'prompt' in item:
        question = item['prompt']
    
    # For answer
    if 'answer' in item:
        answer = item['answer']
    elif 'solution' in item:
        answer = item['solution']
    elif 'ground_truth' in item:
        answer = item['ground_truth']
    
    if question is None or answer is None:
        raise ValueError(f"Cannot find question/answer in item. Available fields: {item.keys()}")
    
    return question, answer

def prepare_training_batch(sampler, questions, solutions_list, device):
    """
    Prepare a batch of training data.
    
    Args:
        sampler: Sampler object with tokenizer
        questions: List of questions
        solutions_list: List of solution lists
        device: Device to move tensors to
        
    Returns:
        input_ids, attention_mask tensors
    """
    # Flatten all question-solution pairs
    texts = []
    for question, solutions in zip(questions, solutions_list):
        for solution in solutions:
            # Format as a complete conversation
            messages = [
                {"role": "system", "content": sampler.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution}
            ]
            text = sampler.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
    
    # Tokenize all texts
    encoded = sampler.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    return encoded['input_ids'].to(device), encoded['attention_mask'].to(device)

def train():
    """Main training function."""
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    print("="*50)
    print("Starting A*-PO Training")
    print("="*50)
    
    # Initialize sampler
    print("\n1. Loading model...")
    sampler = Sampler(
        model_name=config['model_name'],
        device=config['device']
    )
    
    # Create reference model (frozen copy)
    print("2. Creating reference model...")
    ref_model = copy.deepcopy(sampler.model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    # Initialize components
    print("3. Initializing training components...")
    reward_fn = RewardFunction()
    a_star_po = AStarPO(
        model=sampler.model,
        ref_model=ref_model,
        tokenizer=sampler.tokenizer,
        beta=config['beta']
    )
    
    # Optimizer
    optimizer = AdamW(
        sampler.model.parameters(),
        lr=config['learning_rate']
    )
    
    # Load training data
    print("4. Loading training data...")
    train_data = load_jsonl(config['train_data'])
    print(f"   Loaded {len(train_data)} training examples")
    
    # Check data format
    print("\n5. Checking data format...")
    sample_question, sample_answer = get_question_and_answer(train_data[0])
    print(f"   ✓ Question field found: {len(sample_question)} chars")
    print(f"   ✓ Answer field found: {sample_answer[:50]}...")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training Loop")
    print("="*50)
    
    global_step = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        sampler.model.train()
        metrics_tracker = MetricsTracker()
        
        # Process in batches
        batch_size = config['batch_size']
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_data = train_data[start_idx:end_idx]
            
            batch_questions = []
            batch_solutions = []
            batch_rewards = []
            
            # Generate solutions for each question in batch
            for item in batch_data:
                # Use the flexible getter
                question, ground_truth = get_question_and_answer(item)
                
                # Generate multiple solutions
                solutions = sampler.generate_samples(
                    question,
                    num_samples=config['num_samples_per_question'],
                    max_new_tokens=config['max_new_tokens'],
                    temperature=config['temperature'],
                    top_p=config['top_p']
                )
                
                # Calculate rewards
                rewards = reward_fn.calculate_rewards(solutions, ground_truth)
                
                batch_questions.append(question)
                batch_solutions.append(solutions)
                batch_rewards.append(rewards)
                
                # Update metrics
                num_correct = (rewards > 0).sum().item()
                metrics_tracker.update(num_correct, len(solutions))
            
            # Prepare training batch
            input_ids, attention_mask = prepare_training_batch(
                sampler, batch_questions, batch_solutions, config['device']
            )
            
            # Flatten rewards
            rewards_flat = torch.cat(batch_rewards).to(config['device'])
            
            # Compute loss
            loss, loss_metrics = a_star_po.compute_loss(
                input_ids, attention_mask, rewards_flat
            )
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                sampler.model.parameters(),
                config['max_grad_norm']
            )
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            metrics_tracker.update(0, 0, loss=loss.item())
            
            global_step += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"\n  Step {global_step}:")
                print(f"    Loss: {loss_metrics['loss']:.4f}")
                print(f"    Mean Reward: {loss_metrics['mean_reward']:.4f}")
                print(f"    Accuracy: {metrics_tracker.get_accuracy():.4f}")
        
        # End of epoch summary
        summary = metrics_tracker.summary()
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Accuracy: {summary['accuracy']:.4f}")
        print(f"  Mean Reward: {summary['mean_reward']:.4f}")
        print(f"  Mean Loss: {summary['mean_loss']:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(config['output_dir'], f"epoch_{epoch+1}")
        sampler.model.save_pretrained(checkpoint_dir)
        sampler.tokenizer.save_pretrained(checkpoint_dir)
        print(f"  Saved checkpoint to {checkpoint_dir}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

if __name__ == "__main__":
    train()