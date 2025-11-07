# IOZ_HPCTraining

# 🧠 IOZ HPC Training for Students: Using Our DGX / Run:AI Platform

**Dates:** 11–12 November 2025  
**Trainers:** Robin Freeman & Ben Evans  
**Host:** Institute of Zoology, Zoological Society of London (IOZ)  
**Duration:** 1.5 days (Day 1: Core sessions, Day 2: Mini-project + Clinic)

---

## 📋 Overview

This workshop introduces students to the **ZSL High-Performance Computing (HPC) facility**, 

Participants will learn how to access the system, launch workloads, and use CPU and GPU resources for real-world ecological and AI applications.

The course combines short lectures with guided hands-on exercises in R and Python, culminating in a small integrated project using the HPC environment.

---

## 🎯 Learning Objectives

By the end of the workshop, you will be able to:

- Access the Run:AI platform and understand project and storage structure  
- Create and manage CPU and GPU workloads (Jupyter / RStudio sessions)  
- Run parallel computing tasks using multiple cores  
- Execute GPU-accelerated workflows for image analysis  
- Apply HPC methods to **species distribution modelling (SDM)**  
- Use and fine-tune small **large-language models (LLMs)** via Ollama or Hugging Face  
- Design and run your own end-to-end HPC workflow  

---

## 🗓️ Agenda Summary

### **Day 1 – Core HPC Skills**

| Time | Session | Focus |
|------|----------|-------|
| 10:00 – 10:30 | Welcome & Introduction | Overview of DGX systems and Run:AI platform |
| 10:30 – 11:00 | Accounts, Workspaces, Storage & Access | Logins, persistent storage, starting Jupyter/RStudio |
| 11:15 – 11:45 | Session 1: Parallel Computing | Running R/Python code in parallel (CPU) |
| 11:45 – 12:30 | Session 2: GPU Workflows | Image classification with TensorFlow / PyTorch |
| 13:30 – 14:15 | Session 3: SDM | Ecological modelling with `biomod2` / `terra` |
| 14:15 – 15:00 | Session 4: LLMs on the HPC | Running small models via Ollama or Hugging Face |
| 15:15 – 15:45 | Data Management & Reproducibility | FAIR data, containers, and GitHub workflows |

### **Day 2 – Integrated Mini-Project + Clinic**

| Time | Session | Focus |
|------|----------|-------|
| 10:00 – 12:45 | Session 5: Mini-Project | Apply everything learned in an end-to-end workflow |
| 15:00 – 17:00 | Clinic Session | One-to-one help with your own scripts and data |

---

## Mini-Project Options

| Project | Description | Tools |
|----------|--------------|-------|
| **1. Parallel Data Processing Challenge** | Optimise an ecological data aggregation task using parallel computing. | R (`future.apply`) or Python (`multiprocessing`) |
| **2. GPU Image Pipeline Challenge** | Modify or retrain a small CNN on a new image set and benchmark performance. | TensorFlow / PyTorch |
| **3. SDM + LLM Integration Challenge** | Build a species distribution model, then use an LLM to summarise or interpret outputs. | R (`biomod2`, `terra`, `ollamar`) |

---

### Directory structure


---

## Pre-Workshop Setup

- Bring a laptop!
- Basic familiarity with R or Python
- Install: RStudio

---

## Post-Workshop Resources
- “Getting Started with Run:AI” quick-reference guide
- Template notebooks and scripts for each example
- HPC policy and data storage guide
- Teams HPC support channel

---

## Further Reading

- [NVidia Run:AI User Guide](https://run-ai-docs.nvidia.com/)
- [FAIR Data Principles](https://www.go-fair.org/fair-principles/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Ollama Local LLMs](https://ollama.ai/)

---

## Acknowledgements

This workshop was developed by Robin Freeman and Ben Evans, as part of ongoing efforts to build capacity in data-intensive conservation science.

---

© 2025 Zoological Society of London  
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
