# Density_Aware_Greedy_Sampling



## About The Project

This github repo accompanies the paper *A Data Lean Active Learning Architecture for MOF Design Space Exploration* by providing the code used for the implementation of Density-Aware Greedy Smpling  method and the described experiments.



## Abstract

Machine learning algorithms often rely on large training datasets to achieve high performance. However, in domains like chemistry and materials science, acquiring such data is an expensive and laborious process, involving highly trained human experts and material costs. Therefore, it is crucial to develop strategies that minimize the size of training sets while preserving predictive accuracy. The objective is to select an optimal subset of data points from a larger pool of possible samples—one that is sufficiently informative to train an effective machine learning model. Active Learning (AL) methods, which iteratively annotate data points by querying an oracle (e.g., a scientist conducting experiments), have proven highly effective for such tasks. However, challenges remain, particularly for regression tasks, which are generally considered more complex in the AL framework. This complexity stems from the need for uncertainty estimation and the continuous nature of the output space. While both model-free and model-based AL techniques exist to identify new samples for labeling based on unlabeled data, trained models often struggle to accurately capture the conditional relationship between the target response and input features.

In this work, we introduce Density-Aware Greedy Sampling AL (*DAGS-AL*), an active learning method for regression that integrates uncertainty estimation with data density, specifically designed for large design spaces. We evaluate *DAGS-AL* on both synthetic data and multiple real-world datasets of functionalized nanoporous materials, such as MOFs and COFs, for separation applications. Our results demonstrate that *DAGS-AL* consistently outperforms both random sampling and state-of-the-art AL techniques in training regression models effectively with a limited number of data points—even in datasets with a high number of features.



## Configuration

For the implementation of the code we have used *mamba* as our package manager , but *conda* should work fine as well. For specific instructions on installing these package managers please refer to the following links:

- **Conda:**   https://docs.conda.io/en/latest/
- **Mamba:** https://github.com/mamba-org/mamba

After installing your selected package manager you can run the ***env_setup.sh*** bash script contained in the repo. This should create a mamba/conda environment containing all the necessary libraries to execute our code.

The bash script expects two command line arguments. The first one is the name that you want to give to the new environment and the second is whether you are using mamba or conda. So a typical run of the script should look like this:

```bash
./env_setup.sh test_environment mamba
```

After the script has finished just activate the environment by running

```bash
mamba activate test_environment
```

and then you are ready to execute the python file.

## Usage

The code can be run by simply typing 

```bash
python ./main.py
```

in your command line.

Below you will see a series of parameters of the experiments that are configurable and can be selected using command line arguments.

|                  |     CLI Arguments     |                       Possible Inputs                        | Default Inputs |
| ---------------- | :-------------------: | :----------------------------------------------------------: | :------------: |
| Selection Method | -m or <br />--method  |               dags, igs, qbc, rt, random, all                |      all       |
| Dataset          | -d or<br /> --dataset | "forrester", "forrester_imb", "jump_forrester", "jump_forrester_imb", "gaussian", "gaussian_imb", "gaussian_imb_noise", "exponential", "exponential_imb", "all_synthetic"<br />"zifs_diffusivity", "co2", "co2/n2", "o2", "n2", "ch4", "h2", "he", "methane | all_synthetic  |
| Iterations       |     -i or --iter      |                              -                               |       10       |
| Save File Path   |     -s or --save      |                              -                               |  ./ALresults   |



## Data

You can find and download the files containing the real datasets on these links:

[O2 and N2](https://github.com/ibarisorhan/MOF-O2N2/blob/main/mofScripts/MOFdata.csv)

[CH4, H2, He](https://github.com/hdaglar/MOF-basedMMMs_ML/blob/main/rawdata.zip)

Save them in a directory, and remember to use the file path (-p) argument when running the program.

e.g.

```bash
python ./main.py -d o2 -p ./<data_directory>/MOFdata.csv
```

For synthetic datasets you do not need to download anything. The code creates the data.



## License

This project is licensed under the Apache 2 license. See `LICENSE` for details.



## Contact

If you want to contact us you can reach as at the emails of the authors as mentioned in the manuscript.



## Contributors

 <a href= "https://github.com/vGkatsis">Vassilis Gkatsis </a> <br />

 <a href= "https://github.com/PMaratos">Petros Maratos</a> <br />
