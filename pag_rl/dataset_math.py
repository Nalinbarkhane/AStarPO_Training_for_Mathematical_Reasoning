from datasets import load_dataset
import random, re

DATASET_CANDIDATES = [
    # Official repo
    "hendrycks/competition_math",
    # Common mirrors that keep the same fields
    "lighteval/MATH",
    "HuggingFaceH4/MATH",
]

def _extract_answer(solution_text: str):
    if not solution_text:
        return None
    m = re.search(r"\\boxed\{([^}]+)\}", solution_text)
    if m:
        return m.group(1).strip()
    # also accept "Final Answer: ..."
    m = re.search(r"Final Answer:\s*([^\n]+)", solution_text)
    return m.group(1).strip() if m else None

def _postprocess(ds):
    def proc(ex):
        # some mirrors use different keys; normalize to {problem, solution, answer}
        problem = ex.get("problem") or ex.get("question") or ex.get("input") or ""
        solution = ex.get("solution") or ex.get("answer") or ex.get("output") or ""
        return {
            "problem": problem,
            "solution": solution,
            "answer": _extract_answer(solution),
        }
    return ds.map(proc, remove_columns=[c for c in ds.column_names if c not in ("problem", "solution", "answer")])

def load_math_split(split="train"):
    last_err = None
    for ds_id in DATASET_CANDIDATES:
        try:
            ds = load_dataset(ds_id, split=split)
            return _postprocess(ds)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not load any MATH dataset. Tried: {DATASET_CANDIDATES}. "
        f"Last error: {type(last_err).__name__}: {last_err}"
    )

def build_eval_subset(n=500, seed=1337, out_path="data/math500.jsonl"):
    import json, pathlib
    val = load_math_split("test")
    # filter examples with a parsed answer
    val = val.filter(lambda ex: ex.get("answer") is not None)
    random.seed(seed)
    items = random.sample(list(val), k=min(n, len(val)))
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in items:
            f.write(json.dumps({
                "problem": ex["problem"],
                "answer": ex["answer"]
            })+"\n")
