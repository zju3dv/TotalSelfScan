# TotalSelfScan: Learning Full-body Avatars from Self-Portrait Videos of Faces, Hands, and Bodies
### [Project Page](https://zju3dv.github.io/TotalSelfScan/) | [Video](https://www.youtube.com/watch?v=zbNJsqhkees) | [Paper](https://openreview.net/pdf?id=lgj33-O1Ely) | [Data]

> [TotalSelfScan: Learning Full-body Avatars from Self-Portrait Videos of Faces, Hands, and Bodies](https://openreview.net/pdf?id=lgj33-O1Ely)  
> Junting Dong*, Qi Fang*, Yudong Guo, Sida Peng, Qing Shuai, Hujun Bao, Xiaowei Zhou  
> NeurIPS 2022


## training and animation

Please refer to the file `train_pipeline.sh`, where we describe the training and animation pipeline for one example subject.

## Dataset

The dataset can be available [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/12021137_zju_edu_cn/EdhoMoT9JaJFlC3BQrmCWykBErOCx5Q21CXp2mTGikdAJw?e=umAl4n)

```
cd xxx/TotalSelfScan
mkdir data
cd data
mkdir zju_snapshot
mv xxx/male-djt415_tshirt-smplh ./ 
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{dong2022totalselfscan,
  title={TotalSelfScan: Learning Full-body Avatars from Self-Portrait Videos of Faces, Hands, and Bodies},
  author={Dong, Junting and Fang, Qi and Guo, Yudong and Peng, Sida and Shuai, Qing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```