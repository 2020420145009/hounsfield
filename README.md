# Towards Enhancing 3D Spatial Reasoning in Medical Multimodal Large Language Models

This repository contains the official implementation for the paper: **"Towards Enhancing 3D Spatial Reasoning in Medical Multimodal Large Language Models"**.

---

## 📢 News
* **[Coming Soon]** We are currently cleaning up the codebase, organized synthesized reasoning datasets, and evaluation scripts. They will be released as soon as the preparation is complete.

---

## 🎯 Overview

Recent advancements in Multimodal Large Language Models (MLLMs) have shown remarkable success in 2D medical domain, yet extending these capabilities to 3D volumetric data is non-trivial due to severe annotation bottlenecks. 

This work introduces a data-centric paradigm designed to inject explicit clinical reasoning into multi-modal architectures. By transforming global clinical knowledge into fine-grained, slice-wise instructional trajectories, our framework enhances the 3D spatial tracking and differential diagnostic capabilities of foundation models without requiring resource-intensive 3D-specific pre-training.

---

## 🛠️ Repository Roadmap

We plan to release the full project sequentially, including:
- [ ] **Data Synthesis Pipeline:** Scripts for generating structured Chain-of-Thought (CoT) descriptions from volumetric annotations.
- [ ] **Training & Fine-Tuning:** Efficient LoRA fine-tuning recipes for aligning 2D vision-language baselines with 3D clinical logic.
- [ ] **Evaluation Protocols:** LLM-as-a-judge evaluation scripts for parsing 3D spatial coordinate outputs.

---

## ✉️ Contact

For any temporary inquiries regarding the implementation details or collaboration, please open an issue in this repository. 

*(Note: To maintain the integrity of the ongoing academic peer-review process, further institutional contact details will be disclosed upon formal publication.)*
