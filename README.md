<h1 align="center">Shapley Values Evaluation</h1>
<h2 align="center">A Comprehensive Evaluation of Shapley Values Approximations</h2>

## 📄 Contents
1. [Overview](#overview)
2. [Get Started](#start)

   
<h2 id="overview"> 1. Overview </h2>

Understanding the choices made by machine learning models holds significant importance in establishing trust in models' predictions, ultimately facilitating their practical application. The Shapley values have gained popularity as a reliable and theoretically robust approach to foster model interpretability. Shapley values quantify each feature's contribution to model predictions by considering all feature subsets, offering comprehensive insights into their impact. The inherent complexity of computing Shapley values as an NP-hard problem has spurred the development of numerous approximation techniques, leading to an increase in the number of choices in the literature. The abundance of options has created a substantial gap in determining the most appropriate approach for practical applications. Through this study, we seek to bridge this gap by comprehensively evaluating various Shapley value approximation methods. With a fusion of quantitative and qualitative analyses, we rigorously assess the performance and reliability of 17 distinct approximation algorithms across 100 datasets spanning different domains and six different model architectures. Our investigation unveils nuanced insights into the strengths and limitations of each technique. Our evaluation highlights that capturing all the feature interactions is paramount for ensuring accurate and granular model explanations. This study explores different dimensions of Shapley value estimations and ultimately lays the groundwork for developing more reliable and efficient techniques. By leveraging the strengths we identified in existing methods, we aim to motivate further research in model explanations using the Shapley values, fostering continued progress in explainable Artificial Intelligence.

<h2 id="start"> 2. Get Started </h2>


### 2.1 Installation

To install ShapleyValuesEval from source, you will need the following tools:
- `git`
- `conda` (anaconda or miniconda)

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/TheDatumOrg/ShapleyValuesEval.git
```

**Step 2:** Create and activate a `conda` environment named `shapeval`.

```bash
conda env create --file environment.yml
conda activate shapeval
```

