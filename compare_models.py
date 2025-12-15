import os
from pag_rl import Sampler, RewardFunction, load_jsonl, set_seed, load_config

def evaluate_model(model_path, eval_data):
    """Evaluate a model and return accuracy."""
    sampler = Sampler(model_name=model_path, device="cpu")
    reward_fn = RewardFunction()
    
    correct = 0
    total = len(eval_data)
    
    for item in eval_data:
        question = item['problem']
        answer = item['answer']
        
        solution = sampler.generate_single(question, max_new_tokens=200, temperature=0.1)
        if reward_fn.verifier.verify(solution, answer):
            correct += 1
    
    return correct / total

# Load config and data
config = load_config('config.yaml')
eval_data = load_jsonl(config['eval_data'])

print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Base model
print("\n1. Base Model (no training):")
base_acc = evaluate_model(config['model_name'], eval_data)
print(f"   Accuracy: {base_acc:.2%}")

# Epoch 1
print("\n2. After Epoch 1:")
epoch1_acc = evaluate_model("checkpoints/epoch_1", eval_data)
print(f"   Accuracy: {epoch1_acc:.2%}")
print(f"   Improvement: {(epoch1_acc - base_acc):.2%}")

# Epoch 2
print("\n3. After Epoch 2:")
epoch2_acc = evaluate_model("checkpoints/epoch_2", eval_data)
print(f"   Accuracy: {epoch2_acc:.2%}")
print(f"   Improvement: {(epoch2_acc - base_acc):.2%}")

print("\n" + "="*60)
print("BEST MODEL:")
if epoch1_acc >= epoch2_acc:
    print(f"  Epoch 1 ({epoch1_acc:.2%})")
else:
    print(f"  Epoch 2 ({epoch2_acc:.2%})")
print("="*60)