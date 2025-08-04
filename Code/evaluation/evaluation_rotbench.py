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

from utils.utils import load_model, get_params, chat_open, answer_verify, get_feedback, chat_close
from utils.parse_output import get_parse_output
import argparse
import json
import os
from tqdm import trange
from copy import deepcopy
from collections import Counter


def sample_process_open(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "RoTBench"), "model": args.model_path, "metrics": {
        "TS": 0., "PI": 0., "CF": 0.}}

    messages: list = deepcopy(sample['messages'])
    response = chat_open([messages], args)[0]
    messages.append({"role": "assistant", "content": response.strip()})

    answers = []
    for answer in sample['answer']:
        tool = answer.split('Action: ')[1].split('Action Input: ')[0].strip()
        parameters = json.loads(answer.split('Action Input: ')[1].strip())
        assert type(
            parameters) == dict, f"parameters should be dict, but got {type(parameters)}"
        answers.append({"tool": tool, "parameters": parameters.copy()})

    TS, PI, CF = 0., 0., 0.

    try:
        tool = response.split('Action: ')[1].split('Action Input: ')[0].strip()
        parameters = json.loads(response.split('Action Input: ')[1].strip())
        assert type(
            parameters) == dict, f"parameters should be dict, but got {type(parameters)}"

        for answer in answers:
            if answer['tool'] == tool:
                TS = 1.
                if not answer['parameters'].keys() ^ parameters.keys():
                    PI = 1.
                    if Counter(answer['parameters'].values()) == Counter(parameters.values()):
                        CF = 1.
                        break
    except:
        pass

    save_sample["messages"] = deepcopy(messages)
    save_sample["metrics"] = {
        "TS": TS,
        "PI": PI,
        "CF": CF
    }

    return save_sample


def sample_process_close(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "RoTBench"), "model": args.model_path, "metrics": {
        "TS": 0., "PI": 0., "CF": 0.}}

    messages: list = deepcopy(sample['messages'])
    response = chat_close(messages, args)
    if response:
        messages.append(response)

        answers = []
        for answer in sample['answer']:
            tool = answer.split('Action: ')[1].split(
                'Action Input: ')[0].strip()
            parameters = json.loads(answer.split('Action Input: ')[1].strip())
            assert type(
                parameters) == dict, f"parameters should be dict, but got {type(parameters)}"
            answers.append({"tool": tool, "parameters": parameters.copy()})

        TS, PI, CF = 0., 0., 0.

        try:
            tool = response["content"].split(
                'Action: ')[1].split('Action Input: ')[0].strip()
            parameters = json.loads(
                response["content"].split('Action Input: ')[1].strip())
            assert type(
                parameters) == dict, f"parameters should be dict, but got {type(parameters)}"

            for answer in answers:
                if answer['tool'] == tool:
                    TS = 1.
                    if not answer['parameters'].keys() ^ parameters.keys():
                        PI = 1.
                        if Counter(answer['parameters'].values()) == Counter(parameters.values()):
                            CF = 1.
                            break
        except:
            pass

        save_sample["messages"] = deepcopy(messages)
        save_sample["metrics"] = {
            "TS": TS,
            "PI": PI,
            "CF": CF
        }

        return save_sample
    return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default="qwen",
                        choices=["gpt", "qwen", "claude", "gemini"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--input_file", type=str,
                        default="Data/jsonl/raw/RoTBench.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        '--max_turns', type=int, default=1)

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
            ids = [(json.loads(line)["messages"][0]["content"], json.loads(line)["messages"][1]["content"], json.loads(line)["data_source"])
                   for line in f.readlines()]

    if args.series in ["qwen"]:
        args.model, args.tokenizer = load_model(args)
        args.params = get_params(args.series, args)
        args.parse_output = get_parse_output(args.series)

        for i in trange(len(data)):
            sample = data[i]
            if (sample["messages"][0]["content"], sample["messages"][1]["content"], sample["data_source"]) not in ids:
                responses = sample_process_open(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()

    else:
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if (sample["messages"][0]["content"], sample["messages"][1]["content"], sample["data_source"]) not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()


if __name__ == "__main__":
    main()
