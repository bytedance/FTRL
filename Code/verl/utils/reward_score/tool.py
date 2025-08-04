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

from utils.utils import answer_verify, get_feedback
from utils.parse_output import parse_qwen


def compute_solve_pr(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                assert tool_call['id'] == feed['tool_call_id'], f"{tool_call['id']}!= {feed['tool_call_id']}"
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = tmp * tmp / len(parsed_response["tool_calls"])
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")


def compute_solve_rate(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                assert tool_call['id'] == feed['tool_call_id'], f"{tool_call['id']}!= {feed['tool_call_id']}"
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = float(tmp)
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")


def compute_solve_f1(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                assert tool_call['id'] == feed['tool_call_id'], f"{tool_call['id']}!= {feed['tool_call_id']}"
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = 2 * tmp / (len(parsed_response["tool_calls"]) + 1)
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")


def compute_solve_precision(response: str, codes: dict, unsolved_set: dict, solve_rate: float, split: str, answer: str = None):
    unsolved_cnt = sum([len(v) for v in unsolved_set.values()])
    if split == "train":
        response = response.strip().removesuffix(
            '<|endoftext|>').strip().removesuffix('<|im_end|>').strip()
        parsed_response = parse_qwen(response)
        if parsed_response.get('tool_calls', None):
            tmp = 0
            feedback = get_feedback(parsed_response['tool_calls'], codes)
            for tool_call, feed in zip(parsed_response['tool_calls'], feedback):
                assert tool_call['id'] == feed['tool_call_id'], f"{tool_call['id']}!= {feed['tool_call_id']}"
                tool_name = tool_call['function']['name']
                answers = unsolved_set.get(tool_name, [])
                for answer in answers:
                    if answer_verify(feed['content'], answer):
                        answers.remove(answer)
                        tmp += 1
                        break
            score = tmp / len(parsed_response["tool_calls"])
        else:
            if parsed_response['content'] is None or parsed_response['content'] == '':
                score = -0.5
            elif '<tool_call>' in parsed_response['content'] or '</tool_call>' in parsed_response['content'] or '<tool_parsed_response>' in parsed_response['content']:
                score = -0.3
            elif answer is not None and answer.lower().replace(',', '').strip() in parsed_response['content'].lower().replace(',', '').strip():
                score = 1 / (1 + unsolved_cnt)
            elif solve_rate == 1.:
                score = 0.5
            else:
                score = 0.
        return score

    elif split == "test":
        raise NotImplementedError("Not implemented for test split.")

    else:
        raise ValueError(f"Unknown split: {split}")
