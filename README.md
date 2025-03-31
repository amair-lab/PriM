<div align="center">

<div align="center">
 <img src="assets/schema.png" alt="Scalability" style="display:block; margin-left:auto; margin-right:auto;">
   <br>
  <em>
      Overview of the PriM framework
  </em>
</div>

<h3>PriM: Principle-Inspired Material Discovery through Multi-Agent Collaboration</h3>


[Zheyuan Lai](https://zheyuanlai.github.io/)*† and [Yingming Pu](https://dandelionym.github.io/)‡† (Corresponding author)
 
*: National University of Singapore

‡: Westlake University and Zhejiang University

*†: Equal Contribution*

📄 [[PDF]](https://openreview.net/pdf?id=lhobZk76wX) | 💻 [[GitHub]](https://github.com/amair-lab/PriM) | 🌐 [[Homepage]](https://amair-lab.github.io/projects/PriM/)
</div>

<details>
<summary>📖 Abstract</summary>
Complex chemical space and limited knowledge scope with biases holds immense challenge for human scientists, yet in automated materials discovery. Existing intelligent methods relies more on numerical computation, leading to inefficient exploration and results with hard-interpretability. To bridge this gap, we introduce a principles-guided materials discovery workflow powered by language inferential multi-agent system (MAS). Our framework integrates automated hypothesis generation with experimental validation in a roundtable system of MAS, enabling systematic exploration while maintaining scientific rigor. Based on our framework, the case study of nano helix demonstrates higher materials exploration rate and property value while providing transparent reasoning pathways. This approach develops an automated-and-transparent paradigm for material discovery, with broad implications for rational design of functional materials.
</details>

## 👋 Overview
The PriM framework bridges the gap between traditional data-driven methods and principled scientific reasoning. It achieves this by combining:
- **Hypothesis Generation**: A Literature Agent retrieves relevant scientific knowledge, and a Hypothesis Agent formulates testable propositions based on physicochemical principles.
- **Experimental Validation**: An Experiment Agent designs and executes virtual experiments, while an Optimizer Agent refines the parameter space via Monte Carlo Tree Search (MCTS) to maximize the desired material property.
- **Transparent Reasoning**: All decision-making is traceable through explicit, human-readable reasoning paths, facilitating deeper insights into material behavior.

## 📑 Case Study
The system is demonstrated on a **nano helix material discovery** case study, where iterative agent collaboration results in significant improvements in material properties compared to conventional methods.

We set the research goal and constraints as the following:
- **Research Goal (task description)**: Find the structural parameters corresponding to the strongest chirality (g-factor characteristics) in the nanohelix material system.
- **Research Constraints**: Explicitly show the underlying physicochemical principles regarding the structure and property relationships.

<br>
<div align="center">
 <img src="assets/case_study.png" alt="Scalability" style="display:block; margin-left:auto; margin-right:auto;"
   <br>
  <em>
      PriM applied to nano helix material discovery, demonstrating iterative agent collaboration and significant property improvements.
  </em>
</div>
<br>

This figure illustrates the iterative refinement process where hypotheses and experimental conditions are systematically adjusted to optimize the nanohelix's g-factor. 

## 🛠 Setup

1. Clone this repository:
```bash
git clone https://github.com/amair-lab/PriM.git
cd PriM
```

2. Install required dependencies:
```bash
conda create -n PriM python=3.9
conda activate PriM
pip install -r requirements.txt
```

3. Start the Virtual Lab (will be public soon):

4. Run the PriM framework:
```bash
cd src
python inference.py config.yml
```

## 📚 Citation
```bibtex
@inproceedings{pu2025prim,
  title = {PriM: Principle-Inspired Material Discovery through Multi-Agent Collaboration},
  author = {Zheyuan Lai and Yingming Pu},
  booktitle = {ICLR 2025 Workshop on AI for Accelerated Materials Design},
  year = {2025},
  month = mar,
  url = {https://openreview.net/pdf?id=lhobZk76wX},
}
```
