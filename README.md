<h1 align="center">🔍 Shapley Values Evaluation</h1>
<h2 align="center">A Comprehensive Benchmark of Shapley Value Approximations</h2>

<p align="center">
  <img src="./results/replacement/replacementStrategies.png" width="800"/>
</p>

---

## 📄 Contents
1. [Overview](#overview)
2. [Get Started](#get-started)
3. [Evaluation](#evaluation)

---

## 🔎 1. Overview <a name="overview"></a>

Understanding the choices made by machine learning models is essential for building trust and promoting real-world adoption. Shapley values have emerged as a principled and widely-used method for feature attribution. By considering all feature subsets, Shapley values offer comprehensive and fair explanations for model predictions.

However, computing exact Shapley values is computationally intractable (NP-hard), prompting the development of various approximation techniques. The abundance of such methods introduces a new challenge: **Which technique should practitioners trust?**

This work fills that gap through a **systematic and large-scale evaluation** of 17 Shapley value approximation algorithms across:
- 💯 **100 tabular datasets** from diverse domains
- 🧠 **6 model architectures**

We analyze two core aspects:
- **Replacement Strategies** for handling missing features
- **Tractable Estimation Strategies** to approximate Shapley values efficiently

Our results reveal critical trade-offs in accuracy, compute time, and robustness. This benchmark provides the foundation for selecting the right method and encourages further research in interpretable machine learning.

---

## ⚙️ 2. Get Started <a name="get-started"></a>

### ✅ Prerequisites
You will need:
- `git`
- `conda` (Anaconda or Miniconda)

### 📦 Installation

**Step 1:** Clone the repository

```bash
git clone https://github.com/TheDatumOrg/ShapleyValuesEval.git
cd ShapleyValuesEval
```
