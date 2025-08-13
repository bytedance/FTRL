# FTRL
## Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments

> Data for paper [Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](https://arxiv.org/abs/2508.08791)

Junjie Ye

jjye23@m.fudan.edu.cn

Aug. 03, 2025

## Introduction

Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models’ tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models.

<div>
<center>
<img src=Figures/FTRL.png>
</div>

## What's New

- **[2025/08/03]** Release the data and code for FTRL.
- **[2025/08/03]** Paper available on [Arxiv](https://arxiv.org/abs/2508.08791).

## Main Results

We evaluate the performance of various LLMs, and present the average performance across scenarios for each dataset.
Based on it, we make the following observations.
 - Our approach consistently enhances the model's tool-use capabilities across various conditions.
 - Performance gains achieved by our method appear to primarily stem from updates to the model’s lower-layer MLP parameters.
  - Current open-source LLMs do not necessarily exhibit stronger tool-use performance in reasoning mode compared to non-reasoning mode.

<div>
<center>
<img src=Figures/result.png>
</div>

## Usage

### Requirement

- Run the command to install the packages required.
  ```bash
  pip install -r requirements.txt
  ```

### Training with Our Data

- Collected trajectories based on our constructed environments.
  ```bash
  python3 Code/data_sample/data_sample.py --series qwen --model_path ${your_model_path} --input_file Data/jsonl/raw/train.jsonl --output_file ${train_files} --do_sample --temperature 1.0 --device cuda --max_turns 20 --start_id 0 --end_id -1
  ```

- Transform the train data from JSONL format to PARQUET format.
  ```bash
  python3 Code/data_process/tool.py --source_file ${train_files} --overload
  ```

- Train the model with the transformed data.
  ```bash
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.return_raw_chat=False \
    data.train_batch_size=512 \
    data.max_prompt_length=16384 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=messages \
    data.system_style=${system_style} \
    data.enable_thinking=${enable_thinking} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=tool \
    custom_reward_function.path=Code/verl/utils/reward_score/tool.py \
    custom_reward_function.name=compute_solve_f1 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='FTRL' \
    trainer.experiment_name='test' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=120 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
  ```

- Merge the trained model into SATETENSORS format.
  ```bash
  python3 Code/data_process/merge_model.py merge --local_dir ${dir} --target_dir ${dir}
  ```

### Evaluation for Open-Source LLMs

- Run the command to evaluate the Open-Source LLMs. We have supported evaluation for Qwen2.5 and Qwen3 families.
  - Non reasoning mode
    ```bash
    python3 Code/evaluation/${evaluation_file} --series qwen --model_path ${model} --input_file Data/jsonl/raw/${file} --output_file ${save_file} --start_id 0 --end_id -1
    ```
  - Reasoning mode
    ```bash
    python3 Code/evaluation/${evaluation_file} --series qwen --model_path ${model} --input_file Data/jsonl/raw/${file} --output_file ${save_file} --start_id 0 --end_id -1 --enable_thinking
    ```

### Evaluation for Closed-Source LLMs

- Run the command to evaluate the Closed-Source LLMs. We have supported evaluation for Gemini2.5, Claude4.0, and GPT families.
  - Non reasoning mode
    ```bash
    python3 Code/evaluation/${evaluation_file} --series qwen --model_path ${model} --base_url ${base_url} --api_key ${api_key} --input_file Data/jsonl/raw/${file} --output_file ${save_file} --start_id 0 --end_id -1
    ```
  - Reasoning mode
    ```bash
    python3 Code/evaluation/${evaluation_file} --series qwen --model_path ${model} --base_url ${base_url} --api_key ${api_key} --input_file Data/jsonl/raw/${file} --output_file ${save_file} --start_id 0 --end_id -1 --enable_thinking
    ```

## License

The [code](Code) is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgement

We employ the [VeRL](https://arxiv.org/abs/2409.19256) framework for training.

## Citation

If you find this project useful in your research, please cite:

```bibtex
@misc{FTRL,
      title={Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments}, 
      author={Junjie Ye and Changhao Jiang and Zhengyin Du and Yufei Xu and Xuesong Yao and Zhiheng Xi and Xiaoran Fan and Qi Zhang and Xuanjing Huang and Jiecao Chen},
      year={2025},
      eprint={2508.08791},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.08791}, 
}
```
