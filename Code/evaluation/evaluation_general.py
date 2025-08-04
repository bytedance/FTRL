"""
 Copyright 2025 Bytedance Ltd. and/or its affiliates

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import contextlib
from utils.utils import load_model, get_params, chat_open, chat_close
from utils.parse_output import get_parse_output
import argparse
import json
import os
import re
from tqdm import trange
from copy import deepcopy
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
import tempfile
import signal
import os.path as osp
from human_eval.data import HUMAN_EVAL, write_jsonl, read_problems
from human_eval.execution import check_correctness


class TimeOutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _evaluate_mmlu(response_content, sample):
    def _match_answer_pattern(response_text: str, answer_pattern: str):
        match = re.search(answer_pattern, response_text)
        extracted_answer = match.group(1) if match else ''
        return extracted_answer

    answer_pattern = r'(?i)ANSWER\s*:\s*([A-D])'
    pred = _match_answer_pattern(response_content, answer_pattern)
    gold = sample["answer"]

    if pred.lower() == gold.lower():
        return 1.
    else:
        return 0.


def _evaluate_bbh(response_content, sample):
    def _bbh_mcp_postprocess(text: str):
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        match = re.search(r'\(([A-Z])\)*', ans)
        if match:
            return match.group(1)
        match = re.search(r'([A-Z])', ans)
        if match:
            return match.group(1)
        return ans

    def _bbh_freeform_postprocess(text: str):
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        ans = ans.split('\n')[0].strip()
        if ans.endswith('.'):
            ans = ans[:-1].strip()
        match = re.search(r'\*\*(.*?)\*\*', ans)
        if match:
            return match.group(1)
        return ans

    if sample["form"] == "mcp":
        pred = _bbh_mcp_postprocess(response_content)
        gold = _bbh_mcp_postprocess(sample["answer"])
        if pred.lower() == gold.lower():
            return 1.
        else:
            return 0.

    elif sample["form"] == "free":
        pred = _bbh_freeform_postprocess(response_content)
        gold = sample["answer"]
        if pred.lower() == gold.lower():
            return 1.
        else:
            return 0.

    else:
        raise ValueError(f"Unknown form {sample['form']}")


def _evaluate_gsm8k(response_content, sample):
    def _gsm8k_postprocess(text: str):
        text = text.split('Question:')[0]
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
        if not numbers:
            return 'NULL'
        return numbers[-1]

    def _is_equal(pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    pred = _gsm8k_postprocess(response_content)
    gold = sample["answer"]

    if _is_equal(pred, gold):
        return 1.
    else:
        return 0.


def _evaluate_math(response_content, sample):
    gold = sample["answer"]
    gold_with_env = f'${gold}$'
    gold_parsed = parse(
        gold_with_env,
        extraction_mode='first_match',
        extraction_config=[
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
    )

    if len(gold_parsed) != 0:
        pred_parsed = parse(
            response_content,
            extraction_mode='first_match',
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed='all',
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
        )

        if verify(pred_parsed, gold_parsed):
            return 1.

    return 0.


def _evaluate_humaneval(response_content, sample):
    def _humaneval_postprocess_v2(text: str) -> str:
        blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
        if len(blocks) >= 1:
            text = blocks[0]
        return text.lstrip()

    pred = _humaneval_postprocess_v2(response_content)
    gold = sample["answer"]

    problems = read_problems(HUMAN_EVAL)
    args = (problems[gold], pred, 10)
    score = check_correctness(*args)
    
    if score['passed']:
        return 1.
    else:
        return 0.


def _evaluate_mbpp(response_content, sample):
    def _process_answer(text):
        patterns = [
            r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
            r"BEGIN\s*'(.*)'\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)'\s*DONE",
            r"BEGIN\s*'(.*)'\s*DONE",
            r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
            r"BEGIN\s*'(.*)\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)\s*DONE",
            r"BEGIN\s*'(.*)\s*DONE",
            r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
            r'BEGIN\s*(.*)\s*\[DONE\]',
            r'\[BEGIN\]\s*(.*)\s*DONE',
            r'BEGIN\s*(.*)\s*DONE',
            r'```python\s*(.*)\s*```',
            r'```\s*(.*)\s*```',
            r'```python\s*(.*)\s*$',
            r'```\s*(.*)\s*$',
            r'(.*)\s*```.*',
            r"\[BEGIN\]\s*'(.*)",
            r'\[BEGIN\](.*)',
            r"'(.*)'\s*\[DONE\]",
        ]
        for p in patterns:
            match = re.search(p, text, re.DOTALL)
            if match:
                text = match.group(1)
                break
        text = text.split('```')[0]
        text = re.split(r"'?\s*\[?DONE\]?", text)[0]
        text = text.replace('\\_', '_')
        text = text.strip()
        return text

    pred = _process_answer(response_content)
    test_case = sample["answer"]

    programs = pred + '\n' + test_case
    try:
        with time_limit(10):
            exec(programs)
        return 1.
    except TimeOutException:
        return 0.
    except:
        return 0.


def _evaluate(response, sample: dict, args):
    data_source = sample.get("data_source")
    response_content = response["content"]
    if args.enable_thinking:
        if '</think>' in response_content:
            response_content = response_content.split('</think>')[1]

    if data_source == "General/MMLU":
        return _evaluate_mmlu(response_content, sample)
    elif data_source == "General/BBH":
        return _evaluate_bbh(response_content, sample)
    elif data_source == "General/GSM8K":
        return _evaluate_gsm8k(response_content, sample)
    elif data_source == "General/MATH":
        return _evaluate_math(response_content, sample)
    elif data_source == "General/HumanEval":
        return _evaluate_humaneval(response_content, sample)
    elif data_source == "General/MBPP":
        return _evaluate_mbpp(response_content, sample)
    else:
        raise ValueError(f"Unknown data source {data_source}")


def sample_process_open(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "General"), "model": args.model_path, "metrics": {
        "accuracy": 0.}}

    messages: list = deepcopy(sample['messages'])

    response = chat_open([messages], args)[0]
    response = args.parse_output(response)
    messages.append(response)

    save_sample["messages"] = deepcopy(messages)
    save_sample["metrics"] = {
        "accuracy": _evaluate(response, sample, args)
    }

    return save_sample


def sample_process_close(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "General"), "model": args.model_path, "metrics": {
        "accuracy": 0.}}

    messages: list = deepcopy(sample['messages'])

    response = chat_close(messages, args)
    if response:
        messages.append(response.copy())
    else:
        return None

    save_sample["messages"] = deepcopy(messages)
    save_sample["metrics"] = {
        "accuracy": _evaluate(response, sample, args)
    }

    return save_sample


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--input_file", type=str,
                        default="Data/jsonl/raw/General.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    with open(args.input_file, 'r', encoding='utf8') as f:
        data = [json.loads(line) for line in f.readlines()]

    if args.end_id == -1:
        args.end_id = len(data)
    data = data[min(args.start_id, args.end_id): min(args.end_id, len(data))]

    os.makedirs("/".join(args.output_file.split("/")[:-1]), exist_ok=True)

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids = [json.loads(line)["messages"][-2]["content"]
                   for line in f.readlines()]

    if args.series in ["qwen"]:
        args.model, args.tokenizer = load_model(args)
        args.params = get_params(args.series, args)
        args.parse_output = get_parse_output(args.series)

        for i in trange(len(data)):
            sample = data[i]
            if sample["messages"][-1]["content"] not in ids:
                responses = sample_process_open(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()

    else:
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if sample["messages"][-1]["content"] not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()


if __name__ == "__main__":
    main()
