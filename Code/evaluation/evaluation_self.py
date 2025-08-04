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


def sample_process_open(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "self"), "model": args.model_path, "metrics": {
        "solve_precision": 1., "solve_rate": 0., "solve_f1": 0.}}

    messages: list = deepcopy(sample['messages'])
    tools = json.loads(sample['tools'])
    codes = json.loads(sample['codes'])
    unsolved_set: dict = json.loads(sample['unsolved_set'])
    unsolved_cnt = sum([len(item) for item in unsolved_set.values()])
    assert unsolved_cnt > 0, f"unsolved_cnt should be greater than 0, but got {unsolved_cnt}"

    tmp_cnt = 0
    for _ in range(args.max_turns):
        response = chat_open([messages], args, tools)[0]
        response = args.parse_output(response)
        messages.append(response)
        if response.get('tool_calls', None):
            tmp_cnt += len(response["tool_calls"])
            feedback = get_feedback(response['tool_calls'], codes)
            messages.extend(feedback)

            for tool_call, feed in zip(response['tool_calls'], feedback):
                assert tool_call['id'] == feed['tool_call_id'], f"{tool_call['id']}!= {feed['tool_call_id']}"
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        break
        else:
            break

    solved_cnt = unsolved_cnt - sum([len(item)
                                    for item in unsolved_set.values()])
    save_sample["messages"] = deepcopy(messages)
    save_sample["metrics"] = {
        "solve_precision": solved_cnt / tmp_cnt if tmp_cnt > 0 else 1.,
        "solve_rate": solved_cnt / unsolved_cnt,
        "solve_f1": 2 * solved_cnt / (tmp_cnt + unsolved_cnt)
    }

    return save_sample


def sample_process_close(sample: dict, args):
    save_sample = {"messages": [], "data_source": sample.get("data_source", "self"), "model": args.model_path, "metrics": {
        "solve_precision": 1., "solve_rate": 0., "solve_f1": 0.}}

    messages: list = deepcopy(sample['messages'])
    tools = json.loads(sample['tools'])
    codes = json.loads(sample['codes'])
    unsolved_set: dict = json.loads(sample['unsolved_set'])
    unsolved_cnt = sum([len(item) for item in unsolved_set.values()])
    assert unsolved_cnt > 0, f"unsolved_cnt should be greater than 0, but got {unsolved_cnt}"

    tmp_cnt = 0
    for _ in range(args.max_turns):
        response = chat_close(messages, args, tools=tools)
        if response:
            messages.append(response.copy())
            if response.get('tool_calls', None):
                tmp_cnt += len(response["tool_calls"])
                feedback = get_feedback(response['tool_calls'], codes)
                messages.extend(feedback)
                for tool_call, feed in zip(response['tool_calls'], feedback):
                    assert tool_call['id'] == feed[
                        'tool_call_id'], f"{tool_call['id']}!= {feed['tool_call_id']}"
                    tool_name = tool_call['function']['name']
                    answers = unsolved_set.get(tool_name, [])
                    for answer in answers:
                        if answer_verify(feed['content'], answer):
                            answers.remove(answer)
                            break
            else:
                break
        else:
            return None
    solved_cnt = unsolved_cnt - sum([len(item)
                                    for item in unsolved_set.values()])
    save_sample["messages"] = deepcopy(messages)
    save_sample["metrics"] = {
        "solve_precision": solved_cnt / tmp_cnt if tmp_cnt > 0 else 1.,
        "solve_rate": solved_cnt / unsolved_cnt,
        "solve_f1": 2 * solved_cnt / (tmp_cnt + unsolved_cnt)
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
                        default="Data/jsonl/raw/test.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.)
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

    os.makedirs("/".join(args.output_file.split("/")[:-1]), exist_ok=True)

    ids = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf8') as f:
            ids = [json.loads(line)["messages"][0]["content"]
                   for line in f.readlines()]

    if args.series in ["qwen"]:
        args.model, args.tokenizer = load_model(args)
        args.params = get_params(args.series, args)
        args.parse_output = get_parse_output(args.series)

        for i in trange(len(data)):
            sample = data[i]
            if sample["messages"][0]["content"] not in ids:
                responses = sample_process_open(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()

    else:
        args.params = get_params(args.series, args)
        for i in trange(len(data)):
            sample = data[i]
            if sample["messages"][0]["content"] not in ids:
                responses = sample_process_close(sample, args)
                if responses:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(responses, ensure_ascii=False) + '\n')
                        f.flush()


if __name__ == "__main__":
    main()
