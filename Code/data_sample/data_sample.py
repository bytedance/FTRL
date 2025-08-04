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
import json
import os
from func_timeout import func_set_timeout
import json_repair
from random import random
from copy import deepcopy
from utils.utils import answer_verify, load_model, get_params, chat_open, get_feedback
from utils.parse_output import get_parse_output
from tqdm import trange


def sample_process(sample, args):
    save_samples = [deepcopy(sample)]
    messages: list = deepcopy(sample['messages'])
    tools = json.loads(sample['tools'])
    codes = json.loads(sample['codes'])
    unsolved_set: dict = json.loads(sample['unsolved_set'])
    unsolved_cnt = sum([len(item) for item in unsolved_set.values()])

    for _ in range(args.max_turns):
        response = chat_open([messages], args, tools)[0]
        response = args.parse_output(response)
        messages.append(response)
        if response.get('tool_calls', None):
            feedback = get_feedback(response['tool_calls'], codes)
            messages.extend(feedback)

            for tool_call, feed in zip(response['tool_calls'], feedback):
                assert tool_call['id'] == feed['tool_call_id'], f"{tool_call['id']} != {feed['tool_call_id']}"
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        break

            save_samples.append(
                {
                    "messages": deepcopy(messages),
                    "tools": sample["tools"],
                    "codes": sample["codes"],
                    "unsolved_set": json.dumps(unsolved_set, ensure_ascii=False),
                    "solve_rate": 1 - sum([len(item) for item in unsolved_set.values()]) / unsolved_cnt,
                    "data_source": sample["data_source"],
                    "split": sample["split"],
                    "answer": sample["answer"]
                }
            )
        else:
            break

    return save_samples


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["llama31", "qwen"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_file", type=str,
                        default="Data/jsonl/raw/train.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        '--max_turns', type=int, default=20)

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

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids = [json.loads(line)["messages"][0]["content"]
                   for line in f.readlines()]

    args.model, args.tokenizer = load_model(args)
    args.params = get_params(args.series, args)
    args.parse_output = get_parse_output(args.series)

    for i in trange(len(data)):
        sample = data[i]
        if sample["messages"][0]["content"] not in ids:
            responses = sample_process(sample, args)
            if responses:
                with open(args.output_file, 'a') as f:
                    for response in responses:
                        if response:
                            f.write(json.dumps(
                                response, ensure_ascii=False) + '\n')
                            f.flush()


if __name__ == "__main__":
    main()
