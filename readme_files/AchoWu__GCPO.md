# GCPO

> ⚡ Group Contrastive Policy Optimization (GCPO)

[![arXiv](https://img.shields.io/badge/arXiv-2510.07790-b31b1b.svg)](https://arxiv.org/abs/2510.07790)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Ach0/GCPO-R1-1.5B)

**Official repository of the paper:**
**[GCPO: When Contrast Fails, Go Gold](https://arxiv.org/abs/2510.07790)**

---

## About

**GCPO (Group Contrastive Policy Optimization)** is a novel reinforcement learning algorithm designed to enhance the reasoning capabilities of language models, especially in scenarios where the model fails to generate correct responses. Unlike previous methods like GRPO, which rely solely on the model’s own rollouts, GCPO introduces **Golden Answers (GAs)** — external reference answers — to guide the model’s updates when all sampled responses are incorrect.

This approach ensures:

✅ **Full sample utilization** — no training data is wasted  
🧠 **Knowledge transfer** — small models learn reasoning strategies from larger models  
🚀 **Faster convergence** and **better generalization**

---

## 🎯 Key Features

- ✅ **Golden Answer Injection** — handles failure rollouts by injecting correct reference solutions  
- ⚖️ **Sequence-Level Importance Sampling** — stabilizes training under sparse reward settings  
- 🔥 **Contrastive Optimization** — enhances separation between good and bad reasoning traces  
- ✨ **No KL Penalty Needed** — encourages diverse yet effective reasoning behaviors  
- 📚 **Generalizable** — works on math, code, and logical QA tasks

---

## 🚀 Coming Soon

| Item | Status |
|------|--------|
| Paper | ✅ Released |
| Model Checkpoints | ✅ Released |
| GCPO Dataset | ⏳ Coming soon |
| Code (Training + Evaluation) | ⏳ Coming soon |

---

## 🛠️ Model Use
We provide the model weights of **[GCPO-R1-1.5B](https://huggingface.co/Ach0/GCPO-R1-1.5B)**, which is trained based on DeepSeek-R1-Distill-Qwen-1.5B using the GCPO algorithm.
You can find the model at https://huggingface.co/Ach0/GCPO-R1-1.5B.

### ⚖️ Evaluation
To evaluate the model on AIME 2024, run:
```
python3 vllm_eval.py --model_path Ach0/GCPO-R1-1.5B --test_file dataset/AIME24/aime_2024.jsonl --output_path aime2024_result.jsonl  --tensor_parallel_size 4 --mode all
```

---

## 📊 GCPO Improves Reasoning Performance

GCPO consistently outperforms DAPO.

<p align="center">
  <img src="assets/performance.png" alt="Performance Comparison" width="1000">
</p>

---

## 🔧 GCPO Training Pipeline

<p align="center">
  <img src="assets/gcpo_pipeline.png" alt="GCPO Pipeline" width="1200">
</p>

---

## ✍️ Citation

If you find this work useful, please cite:

```bibtex
@article{wu2025gcpo,
  title={GCPO: When Contrast Fails, Go Gold},
  author={Hao Wu and Wei Liu},
  journal={arXiv preprint arXiv:2510.07790},
  year={2025}
}
