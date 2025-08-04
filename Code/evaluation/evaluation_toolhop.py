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

import argparse
from utils.utils import load_model, get_params, chat_open, answer_verify, get_feedback, chat_close
from utils.parse_output import get_parse_output
import json
import os
from tqdm import trange
from copy import deepcopy


def _answer_correct(messages, answer):
    if messages[-1]['role'] == 'assistant':
        if messages[-1]["content"]:
            if answer_verify(messages[-1]["content"], answer):
                return True
            elif messages[-2]["role"] == "tool" and answer_verify(messages[-2]["content"], answer):
                return True
    return False


def sample_process_open_direct(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages', None):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    response = chat_open(messages=[messages], args=args)[0]
    response = args.parse_output(response)
    messages.append(response.copy())
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.

    return save_sample


def sample_process_close_direct(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages', None):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question, and should provide a short answer.\nIf the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the answer is a name, format it as follows: Firstname Lastname\nIf the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the answer in the following format: <answer>your answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    response = chat_close(messages, args)
    if response:
        messages.append(response.copy())
        save_sample['messages'] = messages.copy()
        save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
            messages, sample['answer']) else 0.

        return save_sample
    return None


def sample_process_open_mandatory(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]

    for _ in range(args.max_turns):
        response = chat_open(messages=[messages], args=args, tools=tools)[0]
        response = args.parse_output(response)
        messages.append(response.copy())
        if response.get('tool_calls', None):
            feedback = get_feedback(
                response['tool_calls'], sample['functions'])
            messages.extend(feedback)
        else:
            break
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.

    return save_sample


def sample_process_close_mandatory(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nPlease note that you must call the tool at every step, you must not use your own knowledge. Your final answer must also be returned from the tool.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]

    for _ in range(args.max_turns):
        response = chat_close(messages, args, tools=tools)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                feedback = get_feedback(
                    response['tool_calls'], sample['functions'])
                messages.extend(feedback)
            else:
                break
        else:
            return None
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.
    return save_sample


def sample_process_open_free(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]

    for _ in range(args.max_turns):
        response = chat_open(messages=[messages], args=args, tools=tools)[0]
        response = args.parse_output(response)
        messages.append(response.copy())
        if response.get('tool_calls', None):
            feedback = get_feedback(
                response['tool_calls'], sample['functions'])
            messages.extend(feedback)
        else:
            break
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.

    return save_sample


def sample_process_close_free(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", f"ToolHop/{args.scenario}"), "model": args.model_path, "metrics": {
        "answer_correctness": 0.}}

    if sample.get('messages'):
        messages = sample['messages']
    else:
        messages = [
            {"role": "user", "content": f"You will be asked a question with some tools, and should provide a short final answer.\nIf the final answer is a date, format is as follows: YYYY-MM-DD (ISO standard)\nIf the final nswer is a name, format it as follows: Firstname Lastname\nIf the final answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.\n\nPlease provide the final answer in the following format: <answer>final answer here</answer>\nAnswer as short as possible.\nQuestion: {sample['question']}"}]
    tools = [{"type": "function", "function": tool.copy()}
             for tool in sample["tools"].values()]

    for _ in range(args.max_turns):
        response = chat_close(messages, args, tools=tools)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                feedback = get_feedback(
                    response['tool_calls'], sample['functions'])
                messages.extend(feedback)
            else:
                break
        else:
            return None
    save_sample['messages'] = messages.copy()
    save_sample["metrics"]["answer_correctness"] = 1. if _answer_correct(
        messages, sample['answer']) else 0.
    return save_sample


def sample_process_open(sample: dict, args):
    if args.scenario == "Direct":
        return sample_process_open_direct(sample, args)
    elif args.scenario == "Mandatory":
        return sample_process_open_mandatory(sample, args)
    elif args.scenario == "Free":
        return sample_process_open_free(sample, args)
    else:
        raise NotImplementedError


def sample_process_close(sample: dict, args):
    if args.scenario == "Direct":
        return sample_process_close_direct(sample, args)
    elif args.scenario == "Mandatory":
        return sample_process_close_mandatory(sample, args)
    elif args.scenario == "Free":
        return sample_process_close_free(sample, args)
    else:
        raise NotImplementedError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="Direct",
                        choices=["Direct", "Mandatory", "Free"])
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--input_file", type=str,
                        default="Data/jsonl/raw/ToolHop.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        '--max_turns', type=int, default=9)

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
            ids = [(json.loads(line)["messages"][0]["content"].split('\nQuestion: ')[-1], json.loads(line)["data_source"])
                   for line in f.readlines()]

    if args.series in ["qwen"]:
        args.model, args.tokenizer = load_model(args)
        args.params = get_params(args.series, args)
        args.parse_output = get_parse_output(args.series)

        for i in trange(len(data)):
            sample = data[i]
            if (sample["question"], f"ToolHop/{args.scenario}") not in ids:
                responses = sample_process_open(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()

    else:
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if (sample["question"], f"ToolHop/{args.scenario}") not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()


if __name__ == "__main__":
    main()
