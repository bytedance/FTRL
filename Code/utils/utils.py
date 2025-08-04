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

import json
from func_timeout import func_set_timeout, FunctionTimedOut
from transformers import AutoTokenizer, AutoModelForCausalLM
from hashlib import sha256
from time import sleep
import requests
from copy import deepcopy


def answer_verify(predict, golden,):
    golden = golden.split(', ')
    if type(predict) == dict or type(predict) == list:
        predict = json.dumps(predict, ensure_ascii=False)
    elif type(predict) != str:
        predict = str(predict)
    predict = predict.lower().replace(',', '').strip()

    for item in golden:
        item = item.lower().replace(',', '').strip()
        if item not in predict:
            return False

    return True


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_params(model, args):
    if args.enable_thinking:
        max_new_tokens = 8192
    else:
        max_new_tokens = 1024

    if model == "qwen":
        return dict(do_sample=args.do_sample, max_new_tokens=max_new_tokens, temperature=args.temperature,
                    eos_token_id=[151645, 151643], pad_token_id=args.tokenizer.pad_token_id, use_cache=True)
    elif model == "llama31":
        return dict(do_sample=args.do_sample, max_new_tokens=max_new_tokens, temperature=args.temperature,
                    eos_token_id=[128001, 128008, 128009], pad_token_id=args.tokenizer.pad_token_id, use_cache=True)
    elif model in ["claude", "gemini"]:
        return dict(max_tokens=max_new_tokens, temperature=args.temperature)
    elif model in ["gpt"]:
        return dict(max_tokens=max_new_tokens, temperature=args.temperature)
    else:
        raise NotImplementedError


def chat_open(messages, args, tools=None):
    texts = []
    for message in messages:
        text = args.tokenizer.apply_chat_template(
            message,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        texts.append(text)

    model_inputs = args.tokenizer(
        texts, return_tensors="pt").to(args.device)
    generated_ids = args.model.generate(**model_inputs, **args.params)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = args.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)

    return responses


def chat_close(messages, args, tools=None, one_tool_only=False, base_url=None, api_key=None, model=None, **kwargs):
    def _req_closed():
        try:
            logid = sha256(json.dumps(
                messages[0]["content"]).encode('utf-8')).hexdigest()
            hearers = {
                "Content-Type": "application/json",
                "X-TT-LOGID": logid,
                "caller": "toolfeedback"
            }
            call_model = model if model else args.model_path
            params = deepcopy(args.params)
            if kwargs:
                params.update(kwargs)
            data = {
                "messages": messages,
                "tools": tools,
                "model": call_model,
                **params
            }

            call_base_url = base_url if base_url else args.base_url
            call_api_key = api_key if api_key else args.api_key
            response = requests.post(
                f'{call_base_url}?ak={call_api_key}', headers=hearers, json=data, timeout=50)
            return response.json()
        except Exception as e:
            print(
                f"Warning: There was a promblem when calling close api for messages:\n{messages}\nPass for next.\n{e}")
            return None

    message = {"role": "assistant", "content": ""}
    response = _req_closed()
    if response is not None:
        if response.get("choices"):
            for choice in response['choices']:
                for key, value in choice["message"].items():
                    if value != "":
                        message[key] = value
            if message.get("tool_calls", None):
                if one_tool_only:
                    message["tool_calls"] = message["tool_calls"][:1]
                for tool_call in message["tool_calls"]:
                    if tool_call["function"].get("arguments"):
                        if json.loads(tool_call["function"]["arguments"]) is None:
                            tool_call["function"]["arguments"] = "{}"
                    else:
                        tool_call["function"]["arguments"] = "{}"
        elif "400: Invalid JSON payload received." in response.get("error", {}).get("message", "") or "400: Unable" in response.get("error", {}).get("message", ""):
            return message
        elif "400: Invalid value at" in response.get("error", {}).get("message", ""):
            return message
        elif "The server had an error processing your request." in response.get("error", {}).get("message", ""):
            return message
        elif "is already defined at" in response.get("error", {}).get("message", ""):
            return message
        elif response.get("StopReason", None) == "end_turn":
            return {"role": "assistant", "content": "Sorry, I'm temporarily unable to answer this question."}
        else:
            print(
                f"Warning: There was a promblem when calling close api for messages:\n{messages}\nPass for next.\n{response}")
            return None
    return message


@func_set_timeout(10)
def call_function(name, arguments, code, **kwargs):
    exec(code)
    predict = eval(name)(**arguments, **kwargs)
    if type(predict) == dict or type(predict) == list:
        predict = json.dumps(predict, ensure_ascii=False)
    elif type(predict) != str:
        predict = str(predict)
    return predict


def get_feedback(tool_calls, codes, **kwargs):
    res = []
    for tool_call in tool_calls:
        try:
            tool_name = tool_call['function']['name']
            tool_args = json.loads(tool_call['function']['arguments'])
            code = codes[tool_name]

            feedback = call_function(
                tool_name, tool_args, code, **kwargs)

            res.append({"role": "tool", "content": feedback,
                       "tool_call_id": tool_call["id"]})
        except FunctionTimedOut as e:
            res.append(
                {"role": "tool", "content": f"an error occured when call {tool_call['function']['name']}: {str(e)}", "tool_call_id": tool_call["id"]})
        except Exception as e:
            res.append(
                {"role": "tool", "content": f"an error occured when call {tool_call['function']['name']}: {str(e)}", "tool_call_id": tool_call["id"]})

    return res
