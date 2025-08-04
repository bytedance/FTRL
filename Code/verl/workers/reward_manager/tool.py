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

from collections import defaultdict
import torch
from verl import DataProto
import json


class ToolRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        # the number of batches of decoded responses to print to the console
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            # valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum(
            # )

            # messages = data_item.non_tensor_batch["raw_prompt"]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum(
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum(
            )
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False)

            codes = json.loads(str(data_item.non_tensor_batch["codes"]))
            unsolved_set = json.loads(str(
                data_item.non_tensor_batch["unsolved_set"]))
            solve_rate = data_item.non_tensor_batch["solve_rate"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            split = data_item.non_tensor_batch["split"]
            answer = data_item.non_tensor_batch.get("answer", None)

            score = self.compute_score(
                # messages=messages,
                response=response_str,
                codes=codes,
                unsolved_set=unsolved_set,
                solve_rate=solve_rate,
                split=split,
                answer=answer,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                print("[data_source]", data_source)
                print("[unsolved_set]", unsolved_set)
                # print("[messages]", messages)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[reward_tensor]", reward_tensor[i]
                      [:valid_response_length])
                print("[solve_rate]", solve_rate)
                print("[answer]", answer)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

                already_print_data_sources[data_source] += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
