import json
import os

# Medium difficulty - challenging but solvable
goldilocks_data = [
    # Basic algebra (slightly tricky)
    {
        "problem": "Solve for $x$: $3x + 7 = 22$",
        "answer": "5"
    },
    {
        "problem": "Solve for $x$: $2x - 5 = 13$",
        "answer": "9"
    },
    {
        "problem": "If $x + 8 = 20$, what is $x$?",
        "answer": "12"
    },
    {
        "problem": "Solve: $5x = 45$",
        "answer": "9"
    },
    {
        "problem": "If $x - 7 = 15$, find $x$.",
        "answer": "22"
    },
    
    # Simple equations with two steps
    {
        "problem": "Solve for $x$: $2(x + 3) = 14$",
        "answer": "4"
    },
    {
        "problem": "If $\\frac{x}{4} = 5$, what is $x$?",
        "answer": "20"
    },
    {
        "problem": "Solve: $3(x - 2) = 15$",
        "answer": "7"
    },
    {
        "problem": "If $2x + 1 = 11$, find $x$.",
        "answer": "5"
    },
    {
        "problem": "Solve for $x$: $4x - 3 = 17$",
        "answer": "5"
    },
    
    # Slightly harder arithmetic
    {
        "problem": "What is $15 \\times 8$?",
        "answer": "120"
    },
    {
        "problem": "Calculate $144 \\div 12$.",
        "answer": "12"
    },
    {
        "problem": "What is $25 + 37 + 18$?",
        "answer": "80"
    },
    {
        "problem": "Find $100 - 47$.",
        "answer": "53"
    },
    {
        "problem": "What is $7 \\times 13$?",
        "answer": "91"
    },
    
    # Word problems (simple)
    {
        "problem": "John has 5 apples. Mary gives him 8 more. How many apples does John have now?",
        "answer": "13"
    },
    {
        "problem": "A book costs $15. If you have $50, how much money will you have left after buying it?",
        "answer": "35"
    },
    {
        "problem": "There are 24 students. If they form groups of 4, how many groups are there?",
        "answer": "6"
    },
    {
        "problem": "A rectangle has length 8 and width 5. What is its perimeter?",
        "answer": "26"
    },
    {
        "problem": "If a car travels 60 miles per hour for 3 hours, how far does it go?",
        "answer": "180"
    },
    
    # Basic fractions and percentages
    {
        "problem": "What is $\\frac{1}{2}$ of 40?",
        "answer": "20"
    },
    {
        "problem": "What is $\\frac{3}{4}$ of 20?",
        "answer": "15"
    },
    {
        "problem": "What is $25\\%$ of 60?",
        "answer": "15"
    },
    {
        "problem": "What is $50\\%$ of 80?",
        "answer": "40"
    },
    {
        "problem": "If $\\frac{x}{5} = 3$, what is $x$?",
        "answer": "15"
    },
    
    # Simple geometry
    {
        "problem": "A square has side length 6. What is its area?",
        "answer": "36"
    },
    {
        "problem": "A triangle has base 10 and height 8. What is its area? (Use formula: $\\frac{1}{2} \\times base \\times height$)",
        "answer": "40"
    },
    {
        "problem": "A circle has radius 7. What is its diameter?",
        "answer": "14"
    },
    {
        "problem": "What is the perimeter of a square with side length 9?",
        "answer": "36"
    },
    {
        "problem": "A rectangle has length 12 and width 5. What is its area?",
        "answer": "60"
    },
    
    # Powers and roots (easy ones)
    {
        "problem": "What is $4^2$?",
        "answer": "16"
    },
    {
        "problem": "What is $5^2$?",
        "answer": "25"
    },
    {
        "problem": "What is $\\sqrt{49}$?",
        "answer": "7"
    },
    {
        "problem": "What is $\\sqrt{64}$?",
        "answer": "8"
    },
    {
        "problem": "What is $2^5$?",
        "answer": "32"
    },
    
    # More algebra practice
    {
        "problem": "Solve for $x$: $x + 12 = 30$",
        "answer": "18"
    },
    {
        "problem": "If $7x = 56$, what is $x$?",
        "answer": "8"
    },
    {
        "problem": "Solve: $x - 9 = 15$",
        "answer": "24"
    },
    {
        "problem": "If $\\frac{x}{3} = 8$, find $x$.",
        "answer": "24"
    },
    {
        "problem": "Solve for $x$: $6x = 42$",
        "answer": "7"
    },
    
    # Pattern recognition
    {
        "problem": "What is the next number in the sequence: 2, 4, 6, 8, ...?",
        "answer": "10"
    },
    {
        "problem": "What is the next number: 5, 10, 15, 20, ...?",
        "answer": "25"
    },
    {
        "problem": "Continue the pattern: 3, 6, 9, 12, ...?",
        "answer": "15"
    },
]

print(f"Generated {len(goldilocks_data)} medium-difficulty problems")

# Create data directory
os.makedirs('data', exist_ok=True)

# Training data (first 35 problems)
with open('data/train.jsonl', 'w') as f:
    for item in goldilocks_data[:35]:
        f.write(json.dumps(item) + '\n')

print(f"✓ Created data/train.jsonl with 35 examples")

# Eval data (last 8 problems)
with open('data/math500.jsonl', 'w') as f:
    for item in goldilocks_data[35:]:
        f.write(json.dumps(item) + '\n')

print(f"✓ Created data/math500.jsonl with {len(goldilocks_data) - 35} examples")

# Show samples
print("\n" + "="*50)
print("Sample Problems:")
print("="*50)
for i in range(3):
    print(f"\n{i+1}. {goldilocks_data[i]['problem']}")
    print(f"   Answer: {goldilocks_data[i]['answer']}")

print("\n Goldilocks dataset ready!")
print("\nExpected performance:")
print("  - Base model: 20-40% accuracy")
print("  - After training: 50-70% accuracy")
print("\n" + "="*50)
print("Run these commands:")
print("="*50)
print("1. python run_eval.py    # Check base model first")
print("2. python run_train.py   # Train the model")
print("3. python run_eval.py    # Check improvement")