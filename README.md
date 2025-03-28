# Density_Aware_Greedy_Sampling



## About The Project

This github repo accompanies the paper *A Data Lean Active Learning Architecture for MOF Design Space Exploration (link here)* by providing the code used for the implementation of Density-Aware Greedy Smpling Active Learning method and the corresponding experiments.



## Abstract

Machine learning algorithms often rely on large training datasets to achieve high performance. However, in domains like chemistry and materials science, acquiring such data is an expensive and laborious process, involving highly trained human experts and material costs. Therefore, it is crucial to develop strategies that minimize the size of training sets while preserving predictive accuracy. The objective is to select an optimal subset of data points from a larger pool of possible samples—one that is sufficiently informative to train an effective machine learning model. Active Learning (AL) methods, which iteratively annotate data points by querying an oracle (e.g., a scientist conducting experiments), have proven highly effective for such tasks. However, challenges remain, particularly for regression tasks, which are generally considered more complex in the AL framework. This complexity stems from the need for uncertainty estimation and the continuous nature of the output space. While both model-free and model-based AL techniques exist to identify new samples for labeling based on unlabeled data, trained models often struggle to accurately capture the conditional relationship between the target response and input features.

In this work, we introduce Density-Aware Greedy Sampling AL (*DAGS-AL*), an active learning method for regression that integrates uncertainty estimation with data density, specifically designed for large design spaces. We evaluate *DAGS-AL* on both synthetic data and multiple real-world datasets of functionalized nanoporous materials, such as MOFs and COFs, for separation applications. Our results demonstrate that *DAGS-AL* consistently outperforms both random sampling and state-of-the-art AL techniques in training regression models effectively with a limited number of data points—even in datasets with a high number of features.



## Features

- Feature 1
- Feature 2
- Feature 3



## Configuration

We provide a *requirements.txt* file containing the python libraries required to run our experiments.

As our package manager we have used (and recommend to use) mamba, but conda should work fine as well. For specific instructions on installing these package managers please refer to the following links:

- **Conda:**   https://docs.conda.io/en/latest/
- **Mamba:** https://github.com/mamba-org/mamba





## Usage





## 🧪 Running Tests



## License

This project is licensed under the Apache 2 license. See `LICENSE` for details.



## Contact

If you want to contact us you can reach as at the emails of the authors as mentioned in the manuscript.



## Contributors

 [<img src="https://github.com/vGkatsis.png"  style="border-radius: 50%; border: 1px solid #ccc; width: 60px; height=60px;" /><br /><sub><ahref= "https://github.com/vGkatsis"> Vassilis Gkatsis </a></sub>]

[<img src="https://github.com/PMaratos.png" style="border-radius: 50%; border: 1px solid #ccc; width: 60px; height=60px;"/><br /><sub><ahref= "https://github.com/PMaratos"> Petros Maratos </a></sub>]



## Acknowledgments

- Any contributors
