import os
import argparse
import pandas as pd
import torch
import json
from evaluators.unify_evaluator import unify_Evaluator
import time

# 环境变量设置，允许重复使用库
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    subject_mapping = {
        'applicationsecurity': ['applicationsecurity'],
        'cryptography': ['cryptography'],
        'cybersecurity': ['cybersecurity'],
        'memorysafety': ['memorysafety'],
        'networksecurity': ['networksecurity'],
        'pentest': ['pentest'],
        'softwaresecurity': ['softwaresecurity'],
        'systemsecurity': ['systemsecurity'],
        'vulnerability': ['vulnerability'],
        'websecurity': ['websecurity']
    }

    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    output_dir = args.output_dir
    save_result_dir = os.path.join(output_dir, "results")
    os.makedirs(save_result_dir, exist_ok=True)

    evaluator = unify_Evaluator(choices=args.choices, k=args.ntrain, device=args.device, model_type=args.model_type, model_path=args.model_path, lora_model=args.lora_model, temperature=args.temperature)

    accuracy, summary = {}, {}
    for subject_name in subject_mapping.keys():
        val_file_path = os.path.join('data/val', f'{subject_name}_val.csv')
        if not os.path.exists(val_file_path):
            print(f"Warning: {val_file_path} not found. Skipping...")
            continue

        try:
            val_df = pd.read_csv(val_file_path)
            if val_df.empty:
                print(f"Warning: {val_file_path} is empty or not valid. Skipping...")
                continue
            correct_ratio, all_answers = evaluator.eval_subject(subject_name, val_df, few_shot=args.few_shot, cot=args.cot, save_result_dir=save_result_dir, with_prompt=args.with_prompt, constrained_decoding=args.constrained_decoding, do_test=args.do_test)
            print(f"Subject: {subject_name} - Accuracy: {correct_ratio}")
            accuracy[subject_name] = correct_ratio
            summary[subject_name] = {"score": correct_ratio, "num": len(val_df), "correct": correct_ratio * len(val_df) / 100}
        except Exception as e:
            print(f"An error occurred while processing {val_file_path}: {e}")

    if summary:
        total_num = sum(sub['num'] for sub in summary.values())
        total_correct = sum(sub['correct'] for sub in summary.values())
        summary['All'] = {"score": 100 * total_correct / total_num, "num": total_num, "correct": total_correct}

        print('-' * 80)
        print("Summary of Accuracy by Subject:")
        for k, v in accuracy.items():
            print(f"{k}: {v}")

        print("Average Accuracy Across All Subjects: ")
        print(summary['All']['score'])

        # Save results to JSON files
        json.dump(all_answers, open(os.path.join(save_result_dir, 'submission.json'), 'w'), ensure_ascii=False, indent=4)
        json.dump(summary, open(os.path.join(save_result_dir, 'summary.json'), 'w'), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str)
    parser.add_argument("--cot", type=str2bool, default=False)
    parser.add_argument("--few_shot", type=str2bool, default=False)
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--with_prompt", type=str2bool, default=False)
    parser.add_argument("--constrained_decoding", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n_times", type=int, default=1)
    parser.add_argument("--do_save_csv", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--do_test", type=str2bool, default=False)
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--only_cpu', type=str2bool, default=False, help='Use only CPU for inference')
    parser.add_argument('--choices', nargs='+', help='Choices available for evaluator', required=True)

    args = parser.parse_args()

    args.device = torch.device('cpu' if args.only_cpu else f"cuda:{args.gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus if not args.only_cpu else ""

    for i in range(args.n_times):
        main(args)
