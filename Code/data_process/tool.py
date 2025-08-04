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
import os
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--target_file")
    parser.add_argument("--overload", action="store_true")
    args = parser.parse_args()

    if not args.target_file:
        args.target_file = args.source_file.replace('jsonl', 'parquet')
    if not args.overload and os.path.exists(args.target_file):
        print(f"File {args.target_file} already exists. Nothing to do.")
        return

    with open(args.source_file, "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    df = pd.DataFrame(data)
    df.to_parquet(args.target_file)
    print(f"{len(df)} samples are saved to {args.target_file}.")


if __name__ == "__main__":
    main()
