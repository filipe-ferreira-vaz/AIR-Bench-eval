import json
from collections import defaultdict

input_file = 'Foundation_result_modelx_parsed.jsonl'
output_file = 'accuracy_results.txt'

task_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'invalid': 0})

with open(input_file, 'r') as fp:
    for line in fp:
        data = json.loads(line.strip())
        task_name = data['task_name']
        extracted = str(data['extracted_response']).strip()
        gt = str(data['answer_gt_letter']).strip()

        task_stats[task_name]['total'] += 1
        if extracted not in ('A', 'B', 'C', 'D'):
            task_stats[task_name]['invalid'] += 1
        elif extracted == gt:
            task_stats[task_name]['correct'] += 1

total_all = 0
correct_all = 0
invalid_all = 0
results = []

for task_name, stats in sorted(task_stats.items()):
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    total_all += stats['total']
    correct_all += stats['correct']
    invalid_all += stats['invalid']
    results.append(f"{task_name}: Total={stats['total']}, Correct={stats['correct']}, Invalid={stats['invalid']}, Accuracy={acc:.4f}")

overall_acc = correct_all / total_all if total_all > 0 else 0.0

with open(output_file, 'w') as out:
    out.write("Accuracy Results per Task\n")
    out.write("=" * 70 + "\n")
    for r in results:
        out.write(r + "\n")
    out.write("=" * 70 + "\n")
    out.write(f"OVERALL: Total={total_all}, Correct={correct_all}, Invalid={invalid_all}, Accuracy={overall_acc:.4f}\n")

print(f"Results written to {output_file}")
print(f"\nOverall Accuracy: {overall_acc:.4f} ({correct_all}/{total_all})")
