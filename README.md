# Predicting Drug-Gene Interaction via Multi-Negative Sampling-Based BPR Contrastive Learning


## Citation
If you want to use our code and datasets in your research, please cite:

```
@article{huang2025predicting,
  title={Predicting Drug-Gene Interaction via Multi-Negative Sampling-Based BPR Contrastive Learning},
  author={Huang, Le and Liu, Yongbin and Ouyang, Chunping and Li, Qi and Shan, Jinxiao},
  journal={Tsinghua Science and Technology},
  year={2025},
  publisher={清华大学出版社}
}
```

## 1. Requirements

To run the code, you need the following dependencies:

```
python                        3.10.4
pytorch                       1.11.0
numpy                         1.22.3
pandas                        1.5.2
scipy                         1.7.3
```

## 2. Training and Evaluation

Please use the following command:

* DrugBank
```bash
python Main.py --data DrugBank --iteration 5
```
* DGIdb
```bash
python Main.py --data DGIdb --iteration 5
```
* LINCS L1000
```bash
python Main.py --data LINCS --iteration 1 --load_model ckl/1_model_20230607_DrugBank
```



