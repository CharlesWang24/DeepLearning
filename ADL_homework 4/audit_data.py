import json
import re
from collections import defaultdict
from pathlib import Path


from homework.generate_captions import generate_caption
from homework.generate_qa import generate_qa_pairs

DATA = Path("data")

def parse_image_file(image_file):
    m = re.match(r"(.+)/([^/]+)_(\d+)_im\.jpg", image_file)
    split, base, view = m.group(1), m.group(2), m.group(3)
    return DATA / split / f"{base}_info.jpg", view 

def audit_qa():
    gt = json.load(open(DATA / "valid_grader" / "balanced_qa_pairs.json"))
    total = mismatch = missing = 0
    per_type = defaultdict(lambda: [0,0])
    
    for e in gt:
        info_path, view = parse_image_file(e["image_file"])
        if not info_path.exists():
            continue
        ours = {p["question"]: p["answer"] for p in generate_qa_pairs(str(info_path), view)}
        total += 1 
        template = re.sub(r"\b[a-z_]+\b", "X", e["question"]) if "?" in e["question"] else e["question"]
        q, gt_ans = e["question"], e["answer"].strip().lower()
        if q not in ours:
            missing += 1
            per_type[template][1] += 1
            print(f"Missing: {q} -> {gt_ans}")
            continue
        
        per_type[template][1] += 1
        if ours[q].strip().lower() == gt_ans:
            per_type[template][0] += 1
        else:
            mismatch += 1
            print(f"Mismatch: {q} -> {ours[q]} (expected {gt_ans})")
            
    print(f"Total: {total}, Mismatches: {mismatch}, Missing: {missing}")
    print("Per question type (correct/total):")
    for template, (correct, total) in per_type.items():
        print(f"  {template}: {correct}/{total}")
        
def audit_captions():
    mc = json.load(open(DATA / "valid_grader" / "all_mc_qas.json"))
    hit = total = 0
    for e in mc:
        info_path, view = parse_image_file(e["image_file"])
        if not info_path.exists():
            continue
        ours = set(generate_caption(str(info_path), view))
        correct = e["candidates"][e["correct_index"]]
        total += 1
        if correct in ours:
            hit += 1
        else:
            print(f"Missing: {correct} (ours: {ours})")
    print("Total: {}, Hits: {}, Accuracy: {:.2f}%".format(total, hit, 100 * hit / total))
    
if __name__ == "__main__":
    audit_qa()
    audit_captions()