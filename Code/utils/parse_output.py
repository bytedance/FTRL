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

import json_repair
from hashlib import sha256
import datetime
import json


def parse_qwen(inputs: str, one_tool_only=False):
    output = {"role": "assistant", "content": None, "tool_calls": None}
    start_tool = '<tool_call>'
    end_tool = '</tool_call>'

    start = inputs.find(start_tool)

    if start == -1:
        output['content'] = inputs.strip()
    else:
        output["content"] = inputs[:start].strip() if start != 0 else ''
        tool_calls = inputs[start:]
        tool_calls = tool_calls.split(end_tool)
        if one_tool_only:
            tool_calls = tool_calls[:1]
        for tool_call in tool_calls:
            if tool_call.strip():
                try:
                    if tool_call.strip().startswith(start_tool):
                        tool_call = tool_call.strip().lstrip(start_tool).strip()
                        tool_call = json.loads(tool_call)
                        if tool_call.get("name", None):
                            if type(tool_call['arguments']) != dict:
                                tool_call['arguments'] = json.loads(
                                    tool_call['arguments'])
                            if not output['tool_calls']:
                                output['tool_calls'] = []
                            output['tool_calls'].append({"id": "call_" + sha256(str(datetime.datetime.now()).encode()).hexdigest(), "type": "function", "function": {
                                "arguments": json.dumps(tool_call['arguments'], ensure_ascii=False), "name": tool_call['name']}})
                except Exception as e:
                    print(e)
                    pass
        if output['tool_calls'] is None:
            output['content'] = inputs.strip()
    return output


def parse_llama31(inputs: str):
    output = {"role": "assistant", "content": None, "tool_calls": None}
    start_tool = '<|python_tag|>'

    start = inputs.find(start_tool)

    if start == -1:
        try:
            tool_calls = json_repair.loads(inputs.strip())
            tool_calls = inputs.strip()
        except:
            output['content'] = inputs.strip()
    else:
        tool_calls = inputs[start:].lstrip(start_tool).strip()
        output['content'] = inputs[:start].strip()
    tool_call = tool_calls.split(start_tool)[0]
    tool_call = json_repair.loads(tool_call)
    if type(tool_call) == list:
        tool_call = tool_call[0]

    if tool_call:
        try:
            if type(tool_call['parameters']) != dict:
                tool_call['arguments'] = json_repair.loads(
                    tool_call['parameters'])
            else:
                tool_call['arguments'] = tool_call['parameters'].copy()

            if not output['tool_calls']:
                output['tool_calls'] = []
            output['tool_calls'].append({"id": "call_" + sha256(str(datetime.datetime.now()).encode()).hexdigest(), "type": "function", "function": {
                "arguments": json.dumps(tool_call['arguments'], ensure_ascii=False), "name": tool_call['name']}})
        except Exception as e:
            print(e)
            pass
    if not output['tool_calls']:
        output['content'] = inputs.strip()
    return output


def get_parse_output(model):
    if model == "qwen":
        return parse_qwen
    elif model == "llama31":
        return parse_llama31
    else:
        raise NotImplementedError
