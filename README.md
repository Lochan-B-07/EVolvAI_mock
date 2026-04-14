# EVolvAI: Generative Learning & Evolution Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Autonomous Evolution of Generative Core Architectures.**

EVolvAI is a specialized framework designed to automate the training and architectural evolution of generative AI models. By leveraging an environment-graph based data pipeline, the system dynamically adapts its generative core to optimize for specific output domains, from synthetic data generation to dashboard-driven model monitoring.

## 🚀 The Core Concept: Self-Optimizing Nets
Traditional generative models require manual hyperparameter tuning and architecture design. EVolvAI introduces an evolutionary layer that monitors model performance via a frontend dashboard and feeds structural "mutations" back into the generative core.

## 🛠️ Key Components
- **Generative Core**: The primary model architecture undergoing training and evolution.
- **Environment Graph**: A structured representation of the training data and its relationships, enabling complex context-aware generation.
- **Frontend Dashboard**: Real-time visualization of model fitness, loss curves, and architectural changes.
- **Data Pipeline**: Automated ingestion and sanitization of domain-specific datasets.

---

## 🏗️ Technical Architecture

- **Training Engine**: PyTorch-based training loops with integrated checkpointing.
- **Evolutionary Controller**: Genetic algorithms that optimize model topology.
- **Dashboard API**: Lightweight interface for piping metrics to the frontend.

---

## 📂 Project Structure

- `generative_core/`: Source code for the generative models.
- `environment_graph/`: Logic for the graph-based data representation.
- `data_pipeline/`: Modules for data loading and preprocessing.
- `frontend_dashboard/`: Web-based monitoring interface.
- `Archives/`: Historical roadmaps, training data descriptions, and design docs.

## 🚦 Quick Start

### Installation
```bash
git clone https://github.com/seeramsujay/EVolvAI
cd EVolvAI
pip install -r requirements.txt
```

### Running Training
```bash
# Start the core generative training loop
python run.py
```

### Launch Dashboard
```bash
# Navigate to the dashboard directory and follow the local instructions
cd frontend_dashboard/
```

## 📜 License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
