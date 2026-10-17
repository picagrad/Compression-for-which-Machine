# Attribution and citations

<!-- attribution:begin (generated from the model registry; do not edit by hand) -->
## Attribution

This repository builds on the CFM-Model-Zoo library and on OpenMMLab for the task models, and on CompressAI for the codec's entropy models and layers. The codec, ELFC, adapts ELIC's architecture to feature maps, following Harell et al.'s rate-distortion analysis of coding for machines, and uses QVRF's variable-rate scheme. If you use it, please cite the papers below, and those of the task models you use.

**OpenMMLab:** MMCV (`mmcv`), MMEngine (`mmengine2022`), MMDetection (`mmdetection`), MMSegmentation (`mmseg2020`) and MIM (`mim`).

**Codec:** ELIC (`he2022elic`), Harell et al. (`harell2025rate`), CompressAI (`begaint2020compressai`), QVRF (`tong2023qvrf`) and the CFM-Model-Zoo library (`harell2026cfmmodelzoo`).

**Task models in the experiments:**

| short name | package | cite |
|---|---|---|
| `MaskRCNN` | mmdet | `he2017mask`, `liu2021swin`, `lin2017feature` |
| `RetinaNet` | mmdet | `lin2017focal`, `he2016deep`, `lin2017feature` |
| `CascadeMaskRCNN` | mmdet | `cai2018cascade`, `he2017mask`, `he2016deep`, `lin2017feature` |
| `Mask2FormerSWIN` | mmdet | `cheng2022masked`, `liu2021swin` |
| `Mask2FormerResNet` | mmdet | `cheng2022masked`, `he2016deep` |
| `GroundingDINO` | mmdet | `liu2024grounding`, `liu2021swin`, `devlin2019bert` |
| `GLIP` | mmdet | `li2022grounded`, `zhang2020bridging`, `dai2021dynamic`, `liu2021swin`, `devlin2019bert`, `lin2017feature` |
| `RepPointsV2` | mmdet | `chen2020reppointsv2`, `yang2019reppoints`, `liu2021swin`, `lin2017feature` |
| `ATSSResNet` | mmdet | `zhang2020bridging`, `dai2021dynamic`, `he2016deep`, `lin2017feature` |
| `ATSSSWIN-L` | mmdet | `zhang2020bridging`, `dai2021dynamic`, `liu2021swin`, `lin2017feature` |
| `FCOS` | mmdet | `tian2019fcos`, `zhu2019deformable`, `wu2018group`, `he2016deep`, `lin2017feature` |
| `FasterRCNN` | mmdet | `ren2015faster`, `he2016deep`, `lin2017feature` |
| `DDQDETRResNet` | mmdet | `zhang2023dense`, `zhu2021deformable`, `he2016deep` |
| `DDQDETRSWIN-L` | mmdet | `zhang2023dense`, `zhu2021deformable`, `liu2021swin` |
| `DINOResNet` | mmdet | `zhang2023dino`, `zhu2021deformable`, `he2016deep` |
| `DINOSWIN-L` | mmdet | `zhang2023dino`, `zhu2021deformable`, `liu2021swin` |
| `DeformableDETR` | mmdet | `zhu2021deformable`, `he2016deep` |
| `SegMaskFormerResNet` | mmseg | `cheng2021per`, `he2016deep` |
| `SegMaskFormerSWIN` | mmseg | `cheng2021per`, `liu2021swin` |
| `SegMask2FormerResNet` | mmseg | `cheng2022masked`, `he2016deep` |
| `SegMask2FormerSWIN` | mmseg | `cheng2022masked`, `liu2021swin` |
| `SegKNetUPerResNet` | mmseg | `zhang2021k`, `xiao2018unified`, `he2016deep` |
| `SegKNetSWIN` | mmseg | `zhang2021k`, `xiao2018unified`, `liu2021swin` |
| `SegUperNetResNet` | mmseg | `xiao2018unified`, `he2016deep` |
| `SegUperNetSWIN` | mmseg | `xiao2018unified`, `liu2021swin` |

**BibTeX:**

```bibtex
@misc{mmcv,
  title = {{MMCV}: {OpenMMLab} Computer Vision Foundation},
  author = {{MMCV Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mmcv}},
  year = {2018}
}

@misc{mmengine2022,
  title = {{MMEngine}: {OpenMMLab} Foundational Library for Training Deep Learning Models},
  author = {{MMEngine Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year = {2022}
}

@article{mmdetection,
  title = {{MMDetection}: Open {MMLab} Detection Toolbox and Benchmark},
  author = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal = {arXiv preprint arXiv:1906.07155},
  year = {2019}
}

@misc{mmseg2020,
  title = {{MMSegmentation}: {OpenMMLab} Semantic Segmentation Toolbox and Benchmark},
  author = {{MMSegmentation Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
  year = {2020}
}

@misc{mim,
  title = {{MIM}: {MIM} Installs {OpenMMLab} Packages},
  author = {{MIM Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mim}},
  year = {2021}
}

@inproceedings{he2022elic,
  title = {{ELIC}: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding},
  author = {He, Dailan and Yang, Ziming and Peng, Weikun and Ma, Rui and Qin, Hongwei and Wang, Yan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {5718--5727},
  year = {2022}
}

@article{harell2025rate,
  title = {Rate-Distortion Theory in Coding for Machines and Its Applications},
  author = {Harell, Alon and Foroutan, Yalda and Ahuja, Nilesh and Datta, Parual and Kanzariya, Bhavya and Somayazulu, V. Srinivasa and Tickoo, Omesh and de Andrade, Anderson and Baji{\'c}, Ivan V.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {47},
  number = {7},
  pages = {5501--5519},
  doi = {10.1109/TPAMI.2025.3548516},
  year = {2025}
}

@article{begaint2020compressai,
  title = {{CompressAI}: a {PyTorch} library and evaluation platform for end-to-end compression research},
  author = {B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
  journal = {arXiv preprint arXiv:2011.03029},
  year = {2020}
}

@inproceedings{tong2023qvrf,
  title = {{QVRF}: A Quantization-error-aware Variable Rate Framework for Learned Image Compression},
  author = {Tong, Kedeng and Wu, Yaojun and Li, Yue and Zhang, Kai and Zhang, Li and Jin, Xin},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  pages = {1310--1314},
  year = {2023}
}

@software{harell2026cfmmodelzoo,
  title = {{CFM-Model-Zoo}: a library of split vision models for compression for machines},
  author = {Harell, Alon},
  url = {https://github.com/picagrad/CFM-Model-Zoo},
  version = {0.1.0},
  year = {2026}
}

@inproceedings{he2017mask,
  title = {Mask {R-CNN}},
  author = {He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2017}
}

@inproceedings{liu2021swin,
  title = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2021}
}

@inproceedings{lin2017feature,
  title = {Feature Pyramid Networks for Object Detection},
  author = {Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}

@inproceedings{lin2017focal,
  title = {Focal Loss for Dense Object Detection},
  author = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2017}
}

@inproceedings{he2016deep,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016}
}

@inproceedings{cai2018cascade,
  title = {Cascade {R-CNN}: Delving into High Quality Object Detection},
  author = {Cai, Zhaowei and Vasconcelos, Nuno},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}

@inproceedings{cheng2022masked,
  title = {Masked-attention Mask Transformer for Universal Image Segmentation},
  author = {Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and Kirillov, Alexander and Girdhar, Rohit},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}

@inproceedings{liu2024grounding,
  title = {Grounding {DINO}: Marrying {DINO} with Grounded Pre-Training for Open-Set Object Detection},
  author = {Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Jiang, Qing and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}

@inproceedings{devlin2019bert,
  title = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle = {Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019}
}

@inproceedings{li2022grounded,
  title = {Grounded Language-Image Pre-training},
  author = {Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei and Hwang, Jenq-Neng and Chang, Kai-Wei and Gao, Jianfeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}

@inproceedings{zhang2020bridging,
  title = {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author = {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

@inproceedings{dai2021dynamic,
  title = {Dynamic Head: Unifying Object Detection Heads with Attentions},
  author = {Dai, Xiyang and Chen, Yinpeng and Xiao, Bin and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Zhang, Lei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}

@inproceedings{chen2020reppointsv2,
  title = {{RepPoints} v2: Verification Meets Regression for Object Detection},
  author = {Chen, Yihong and Zhang, Zheng and Cao, Yue and Wang, Liwei and Lin, Stephen and Hu, Han},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020}
}

@inproceedings{yang2019reppoints,
  title = {{RepPoints}: Point Set Representation for Object Detection},
  author = {Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{tian2019fcos,
  title = {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author = {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{zhu2019deformable,
  title = {Deformable {ConvNets} v2: More Deformable, Better Results},
  author = {Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@inproceedings{wu2018group,
  title = {Group Normalization},
  author = {Wu, Yuxin and He, Kaiming},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}

@inproceedings{ren2015faster,
  title = {Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
  author = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2015}
}

@inproceedings{zhang2023dense,
  title = {Dense Distinct Query for End-to-End Object Detection},
  author = {Zhang, Shilong and Wang, Xinjiang and Wang, Jiaqi and Pang, Jiangmiao and Lyu, Chengqi and Zhang, Wenwei and Luo, Ping and Chen, Kai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}

@inproceedings{zhu2021deformable,
  title = {Deformable {DETR}: Deformable Transformers for End-to-End Object Detection},
  author = {Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2021}
}

@inproceedings{zhang2023dino,
  title = {{DINO}: {DETR} with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author = {Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel M. and Shum, Heung-Yeung},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}

@inproceedings{cheng2021per,
  title = {Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author = {Cheng, Bowen and Schwing, Alexander G. and Kirillov, Alexander},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}

@inproceedings{zhang2021k,
  title = {{K-Net}: Towards Unified Image Segmentation},
  author = {Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Loy, Chen Change},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}

@inproceedings{xiao2018unified,
  title = {Unified Perceptual Parsing for Scene Understanding},
  author = {Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}
```
<!-- attribution:end -->
