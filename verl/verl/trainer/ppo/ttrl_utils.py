# # Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# from typing import List
# from collections import Counter
# import torch
# import numpy as np
# from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

# def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
#     """
#     Select the first k rollouts per prompt, used for TTRL downsampling.
#     """
#     assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
#     num_prompts = len(data) // n_votes_per_prompt

#     selected_indices = []
#     for i in range(num_prompts):
#         start = i * n_votes_per_prompt
#         selected_indices.extend(range(start, start + n_samples_per_prompt))

#     return data[selected_indices]


# # === Ground Truth Manipulation ===


# def apply_original_gt(batch):
#     """
#     Apply the original ground truth to the batch.
#     """
#     for i in range(len(batch)):
#         data_item = batch[i]
#         original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
#         data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

#     return batch


# def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
#     """
#     Apply the majority vote ground truth to the batch.
#     """
#     assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
#     num_prompts = len(gen_batch_output) // n
#     assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

#     model_outputs = []  
#     for i in range(num_prompts):
#         start = i * n
#         for j in range(n):
#             data_item = gen_batch_output[start + j]
#             prompt_ids = data_item.batch["prompts"]
#             prompt_length = prompt_ids.shape[-1]
#             response_ids = data_item.batch["responses"]
#             valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]
#             response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
#             model_outputs.append(response_str)

#     majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
#     assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
#     for i in range(num_prompts):
#         data_item = batch[i]
#         original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
#         data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
#         data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
#         data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

#     batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
#     return batch


# def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
#     """
#     Used to generate the ground truth for TTRL.
#     Input:
#         model_outputs: list of str
#         n: int
#     Output:
#         majority_gt_list: list of str
#         majority_ratio_list: list of float
#     """
#     majority_gt_list = []
#     majority_ratio_list = []
#     assert len(model_outputs) % n == 0
#     n_prompts = len(model_outputs) // n
#     for i in range(n_prompts):
#         prompt_outputs = model_outputs[i * n:(i + 1) * n]
#         prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
#         majority_gt_list.append(prompt_majority_gt)
#         majority_ratio_list.append(prompt_majority_ratio)
        
#     return majority_gt_list, majority_ratio_list


# def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
#     assert len(model_outputs) > 0
#     model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
#     model_answers = [answer for answer in model_answers if answer is not None]
#     model_answers = [simplify_expression_string(answer) for answer in model_answers]
#     if len(model_answers) == 0:
#         return "None", 0.0
    
#     counter = Counter(model_answers)
    
#     majority_answer, majority_count = counter.most_common(1)[0]
#     majority_ratio = majority_count / len(model_outputs)
    
#     return majority_answer, majority_ratio


# # === Metrics Computation ===


# def compute_ttrl_metrics(batch, n):
#     """
#     Compute the TTRL metrics.
#     """
#     assert len(batch) % n == 0, "batch length must be divisible by n"
#     num_prompts = len(batch) // n

#     # Sort the batch by the ID
#     idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

#     majority_reward = []
#     gt_reward = []
#     majority_label = []
#     gt_label = []

#     for i in range(len(batch)):
#         data_item = batch[idx[i]]
#         majority_reward.append(data_item.batch["token_level_scores"].sum().item())
#         gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
#         majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
#         gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

#     ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
#     majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
#     majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
#     ttrl_metrics["majority_ratio"] = majority_ratio

#     return ttrl_metrics


# def _batch_compute_ttrl_metrics(
#     majority_reward: List[float],
#     gt_reward: List[float],
#     majority_label: List[str],
#     gt_label: List[str],
#     n: int,
# ):
#     """
#     Compute the TTRL metrics for batch inputs.
#     """
#     assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
#     assert len(majority_reward) % n == 0
#     n_prompts = len(majority_reward) // n
#     ttrl_metrics = []
#     for i in range(n_prompts):
#         prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
#         prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
#         prompt_majority_label = majority_label[i * n:(i + 1) * n]
#         prompt_gt_label = gt_label[i * n:(i + 1) * n]

#         assert Counter(prompt_majority_label).most_common(1)[0][1] == n
#         assert Counter(prompt_gt_label).most_common(1)[0][1] == n

#         prompt_majority_label = prompt_majority_label[0]
#         prompt_gt_label = prompt_gt_label[0]

#         ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
#         ttrl_metrics.append(ttrl_metric)

#     # Compute the average metrics
#     ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

#     return ttrl_metrics

# def _prompt_compute_ttrl_metrics(
#     majority_reward: List[float],
#     gt_reward: List[float],
#     majority_label: str,
#     gt_label: str,
#     ):    
#     assert len(majority_reward) == len(gt_reward)

#     hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
#     rewards_hit_rate = 0
#     for estimate_reward, true_reward in zip(majority_reward, gt_reward):
#         if estimate_reward == true_reward:
#             rewards_hit_rate += 1
#     rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
#     ttrl_metric = {
#         "label_accuracy": hit_rate,
#         "reward_accuracy": rewards_hit_rate,
#         "majority_voting_reward": sum(majority_reward) / len(majority_reward),
#         "ground_truth_reward": sum(gt_reward) / len(gt_reward),
#         f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
#     }
#     return ttrl_metric


# Copyright 2025 TTRL Team (Modified for SASC Algorithm)
from typing import List, Dict
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]

# === Ground Truth Manipulation ===

def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
    return batch

def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the SASC weighted vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []
    stability_scores = [] # Store stability scores

    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            # Valid response length calculation
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)
            
            # Extract stability score from meta_info (Populated by vLLMRollout)
            # Default to 100.0 (high instability) if not found to avoid crashing
            score = data_item.non_tensor_batch.get("stability_score", 100.0)
            if isinstance(score, np.ndarray): 
                score = score.item()
            stability_scores.append(float(score))

    # 使用 SASC 算法进行加权投票
    majority_gt_list, majority_ratio_list = _batch_sasc_vote(model_outputs, stability_scores, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch

def _batch_sasc_vote(model_outputs: List[str], stability_scores: List[float], n: int) -> tuple[List[str], List[float]]:
    """
    SASC Algorithm Implementation: FilterTopK + W-StdTopK
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    assert len(stability_scores) == len(model_outputs)
    
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        start = i * n
        end = (i + 1) * n
        prompt_outputs = model_outputs[start:end]
        prompt_stabilities = stability_scores[start:end]
        
        # 调用核心 SASC 逻辑
        prompt_majority_gt, prompt_majority_ratio = _sasc_weighted_vote(prompt_outputs, prompt_stabilities)
        
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list

def _sasc_weighted_vote(model_outputs: List[str], stability_scores: List[float]) -> tuple[str, float]:
    """
    SASC Core Logic:
    1. Filter Unstable (Stability > P80)
    2. Weighted Vote by Exp(-Z_score of Stability)
    """
    if len(model_outputs) == 0:
        return "None", 0.0

    # 1. 预处理：解析答案
    parsed_answers = []
    valid_indices = []
    
    for idx, output in enumerate(model_outputs):
        ans = extract_answer(output)
        if ans is not None:
            ans = simplify_expression_string(ans)
            parsed_answers.append(ans)
            valid_indices.append(idx)
    
    if not valid_indices:
        # Fallback to simple majority vote on raw strings if extraction fails
        # Or return None. Here we return None to be safe.
        return "None", 0.0

    current_stabilities = np.array([stability_scores[i] for i in valid_indices])
    current_answers = parsed_answers 

    # === Stage 1: Filter Unstable (过滤掉 Stability 最大的 20%) ===
    # 注意：Stability (Std) 越小越好。剔除大的。
    if len(current_stabilities) >= 5: # 样本太少不过滤
        threshold_p80 = np.percentile(current_stabilities, 80)
        # 保留 stability <= P80 的样本 (即保留较小的80%)
        mask = current_stabilities <= threshold_p80
        
        # 再次检查过滤后是否为空（理论上不会，除非所有值都NaN）
        if np.sum(mask) > 0:
            filtered_answers = [current_answers[i] for i in range(len(current_answers)) if mask[i]]
            filtered_stabilities = current_stabilities[mask]
        else:
            filtered_answers = current_answers
            filtered_stabilities = current_stabilities
    else:
        filtered_answers = current_answers
        filtered_stabilities = current_stabilities

    # === Stage 2: Weighted Voting (Exp(-Z)) ===
    # 权重计算：越稳定（值越小），Z越小，-Z越大，Exp(-Z)越大 -> 权重越大
    if len(filtered_stabilities) > 1 and np.std(filtered_stabilities) > 1e-9:
        mu = np.mean(filtered_stabilities)
        sigma = np.std(filtered_stabilities)
        z_scores = (filtered_stabilities - mu) / sigma
        weights = np.exp(-z_scores)
    else:
        # 只有一个样本，或所有样本稳定性完全相同
        weights = np.ones(len(filtered_stabilities))

    # 加权聚合
    weighted_votes = {}
    total_weight = np.sum(weights) + 1e-9
    
    for ans, w in zip(filtered_answers, weights):
        weighted_votes[ans] = weighted_votes.get(ans, 0.0) + w
        
    # 选出得分最高的
    best_answer = max(weighted_votes, key=weighted_votes.get)
    # 计算加权置信度 (Weighted Confidence)
    best_ratio = weighted_votes[best_answer] / total_weight
    
    return best_answer, best_ratio

# === Metrics Computation ===

def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []
    
    # [ADDED] List to collect stability scores for WandB logging
    all_stability_scores = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 
        
        # [ADDED] Extract stability score
        if "stability_score" in data_item.non_tensor_batch:
            s_score = data_item.non_tensor_batch["stability_score"]
            if hasattr(s_score, 'item'):
                all_stability_scores.append(s_score.item())
            else:
                all_stability_scores.append(float(s_score))

    # Note: metrics calculation doesn't strictly depend on the voting method internals,
    # as long as majority_label is populated correctly.
    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    # [ADDED] Add stability metrics to the return dictionary
    if len(all_stability_scores) > 0:
        ttrl_metrics["stability_mean"] = sum(all_stability_scores) / len(all_stability_scores)
        ttrl_metrics["stability_min"] = min(all_stability_scores)
        ttrl_metrics["stability_max"] = max(all_stability_scores)

    return ttrl_metrics

def _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n):
    assert len(majority_reward) == len(gt_reward)
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        start = i * n
        end = (i + 1) * n
        prompt_metrics = _prompt_compute_ttrl_metrics(
            majority_reward[start:end], 
            gt_reward[start:end], 
            majority_label[start], 
            gt_label[start]
        )
        ttrl_metrics.append(prompt_metrics)
    
    final_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}
    return final_metrics

def _prompt_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label):
    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric