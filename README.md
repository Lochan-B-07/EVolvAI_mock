# EVolvAI: Generative Counterfactual Framework

## Overview
This repository contains the core deep learning architecture for the **Generative Counterfactual Framework** – a crucial component of the EVolvAI project. 

The traditional literature in power systems forecasting heavily relies on deterministic predictions based on historical behavior (e.g., LSTMs and standard RNNs). However, standard models cannot extrapolate to extreme "what-if" scenarios crucial for modern grid resilience and planning.

This module bridges that gap. By combining **Temporal Convolutional Networks (TCNs)** and **Variational Autoencoders (VAEs)** with **intervention-based latent conditioning**, this system goes beyond mere forecasting. It actively generates realistic, high-fidelity 24-hour demand profiles specifically mapped to physical grid topology under extreme or unseen circumstances (e.g., severe winter storms coupled with 100% EV fleet electrification surges).

---

## 🚀 What Has Been Completed (Role: Generative AI Modeler)

1. **Theoretical Foundation (`sector_3_review.md`)**
   - Drafted the "Sector 3" literature review highlighting the limitations of current forecasting models.
   - Justified the use of Causal Machine Learning and Conditional VAEs to power intervention-based demand profiling.

2. **Async Handoff System (`mock_output.py`)**
   - Implemented a temporary Python generator that mimics the expected tensor output shape (`[24_hours, number_of_nodes]`).
   - *Status:* **Ready for immediate use.** This unblocks the Grid Physics and Optimization teams while the heavy PyTorch models finish their long training cycles.

3. **Core Model Architecture (`models.py` & `data_loader.py`)**
   - Built custom `EVDemandDataset` loaders to handle both historical charging behavior and exogenous weather variables.
   - Built the `GenerativeCounterfactualVAE` using PyTorch.
   - Utilizes `CausalConv1d` (TCNs) in the encoder and decoder to synthesize 24-hour temporal distributions without vanishing gradient issues.
   - Specifically engineered the latent space to accept condition vectors (`[WeatherFlag, EV_Multiplier]`) for counterfactual scenario generation.

4. **Remote GPU Training Pipeline (`EVolvAI_Training.ipynb`)**
   - Packaged the entire PyTorch architecture, loaders, and an active training loop into a single Google Colab compatible Jupyter Notebook.
   - Allows for immediate execution and training on remote cloud GPUs.

---

## ⏭️ Next Steps for the Pipeline

### Action Items for Grid Physics (Akshay) & Optimization (Teammate B)

Because the project workflow has been designed asynchronously, **you do not need to wait for the deep learning models to finish training.**

**Step 1: Use the Mock Data Immediately**
1. Run `python mock_output.py`.
2. This will generate a mathematically sound `[24, N]` numpy array (`mock_demand_tensor.npy`) representing demand in kW across the nodes for a full day.
3. **Akshay (Grid Physics):** Pipe this tensor directly into your `pandapower` or `MATPOWER` environments. Begin mapping this demand to your local transformer nodes, running power flow simulations, and coding up the penalty engines for voltage drops or line thermal overloads.
4. **Teammate B (Optimization):** Use the outcome of the grid penalties to begin building your Genetic Algorithm or optimization heuristics targeting those bottlenecks.

**Step 2: Swap to Real AI Output (Final Integration)**
1. The AI Modeler will run the `EVolvAI_Training.ipynb` notebook on Google Colab to train the model on large datasets.
2. Once the model effectively learns the baseline distribution, counterfactual condition vectors (e.g., `[1.0, 2.5]`) will be injected into the latent space to generate an entirely novel, physics-driven surge scenario.
3. The AI Modeler will provide you with the *final* `[24, N]` tensor output from this trained model.
4. Simply replace `mock_demand_tensor.npy` in your code with the new AI-generated tensor. Because the output structure is strictly identical, your grid simulation and optimization code will run perfectly without any modifications.

---

## Installation & Local Testing

If you need to test the PyTorch architecture locally:

```bash
pip install torch numpy
python build_colab_notebook.py  # Generates the latest Colab Notebook
```
