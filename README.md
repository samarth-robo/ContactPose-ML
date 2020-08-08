# [ContactPose](https://contactpose.cc.gatech.edu)

Code for the **data-driven hand-object contact modeling** experiments presented in the paper:

[ContactPose: A Dataset of Grasps with Object Contact and Hand Pose]() - 

[Samarth Brahmbhatt](https://samarth-robo.github.io/),
[Chengcheng Tang](https://scholar.google.com/citations?hl=en&user=WbG27wQAAAAJ),
[Christopher D. Twigg](https://scholar.google.com/citations?hl=en&user=aN-lQ0sAAAAJ),
[Charles C. Kemp](http://charliekemp.com/), and
[James Hays](https://www.cc.gatech.edu/~hays/),

**ECCV 2020**.

Please visit [http://contactpose.cc.gatech.edu](http://contactpose.cc.gatech.edu) to explore the dataset.

## Citation
```
@InProceedings{Brahmbhatt_2020_ECCV,
author = {Brahmbhatt, Samarth and Tang, Chengcheng and Twigg, Christopher D. and Kemp, Charles C. and Hays, James},
title = {{ContactPose}: A Dataset of Grasps with Object Contact and Hand Pose},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```

# Getting Started

- Clone this repository
```bash
$ git clone git@github.com:samarth-robo/ContactPose-ML.git contactpose-ml
$ cd contactpose-ml
```
-  Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Create the `contactpose_ml` conda environment:
`conda env create -f environment.yml`. Activate it:
```bash
$ source activate contactpose_ml
```
- Checkout the appropriate branch for the features used to predict contact:
  - [simple-joints](https://github.com/samarth-robo/ContactPose-ML/tree/simple-joints)
  - [relative-joints](https://github.com/samarth-robo/ContactPose-ML/tree/relative-joints)
  - [skeleton](https://github.com/samarth-robo/ContactPose-ML/tree/skeleton)
  - [mesh](https://github.com/samarth-robo/ContactPose-ML/tree/mesh)
  - [images](https://github.com/samarth-robo/ContactPose-ML/tree/images)

# Download Links

## Trained Models

| **Learner**         | **Features**    | **Split**   | **Link** |
|---------------------|-----------------|-------------|------|
| MLP | simple-joints | object | [link](https://www.dropbox.com/sh/diu3ceafm2d29f7/AAB23ugU_1oWQ1kk6lNAKZyya?dl=1) |
| MLP | relative-joints | object | [link](https://www.dropbox.com/sh/ifb37j6h8ni8851/AAA7JLz96cKmZwzTRwI6G9Vza?dl=1)|
| MLP | skeleton | object | [link](https://www.dropbox.com/sh/jszbnc5txyp1lny/AABKT8Z9DeHlgZyP7hRxilToa?dl=1) |
| MLP | mesh | object | [link](https://www.dropbox.com/sh/4dt5rk8ker3hx56/AABni1Czr6RfQ_6r4pTf18oWa?dl=1) |
| MLP | simple-joints | participant | [link](https://www.dropbox.com/sh/4im9mm4nluy5vna/AADOGgTwVClXfmLSojhgDfZYa?dl=1) |
| MLP | relative-joints | participant | [link](https://www.dropbox.com/sh/0ztxdonvdhbftoj/AACJCU3FLQFMo9BINwIc-ZLFa?dl=1) |
| MLP | skeleton | participant | [link](https://www.dropbox.com/sh/x6wl7pbj64y3zxa/AAASD_pFlaUVtgD6_lLIO2lQa?dl=1) |
| MLP | mesh | participant | [link](https://www.dropbox.com/sh/z5q92scdcm4vz41/AABH88OJOFzvAG47y5i9HqNva?dl=1) |
| PointNet++ | simple-joints | object | [link](https://www.dropbox.com/sh/osq52js7v67f86w/AACiTAWVfiYCo5sqh6LLNLTya?dl=1) |
| PointNet++ | relative-joints | object | [link](https://www.dropbox.com/sh/6qzsu7dfrw29qzn/AABbiLDaKGz0g06xe25cuJEza?dl=1) |
| PointNet++ | skeleton | object | [link](https://www.dropbox.com/sh/oo2xsjoklnxwfi7/AAB1By9ELXgpcxfxu11zvKFka?dl=1) |
| PointNet++ | mesh | object | [link](https://www.dropbox.com/sh/sbskyrjffansvkn/AADIoqzUOv2lh8kzUPzdQkNBa?dl=1) |
| PointNet++ | simple-joints | participant | [link](https://www.dropbox.com/sh/6p3incblwvf87e8/AAATyVHlddD-sEURHoS8Dn-ya?dl=1) |
| PointNet++ | relative-joints | participant | [link](https://www.dropbox.com/sh/ejfljimt4dj92tu/AADq-cPe3nFQ5cXXpQHTlO_Wa?dl=1) |
| PointNet++ | skeleton | participant | [link](https://www.dropbox.com/sh/e6gm4mwsbaqml89/AACPnfzQtL1-YqoMpXkIiA4na?dl=1) |
| PointNet++ | mesh | participant | [link](https://www.dropbox.com/sh/rx4pge4m6mpsb86/AACtSocTvxOexVl3VFL_p8Xpa?dl=1) |
| VoxNet | skeleton | object | [link](https://www.dropbox.com/sh/1ykqz8pddya1zu7/AABXAuwMIaBLncLmq2t_hoRIa?dl=1) |
| VoxNet | skeleton | participant | [link](https://www.dropbox.com/sh/13mygnw3yu70f5u/AADIGFoV_HNBvoRc_iRGypURa?dl=1) |
| Heuristic (10 pose params) | - | object | [link](https://www.dropbox.com/sh/478b5v3gp6euzom/AABt_-24TBlglf3c_mctklwZa?dl=1) |
| Heuristic (15 pose params) | - | object | [link](https://www.dropbox.com/sh/8zidjcmxp50cpuu/AAC6Gq4Kx_AwtOREd6Nh5Of_a?dl=1) |
| Heuristic (10 pose params) | - | participant | [link](https://www.dropbox.com/sh/l1erk9cm3h740st/AADp3MG1-L-PdBH6k11v8fxsa?dl=1) |
| Heuristic (15 pose params) | - | participant | [link](https://www.dropbox.com/sh/lob3yszezysj6ni/AADKhDrOhtJGNuqRETBhFds8a?dl=1) |
| enc-dec, PointNet++ | images (3 view) | object | [link](https://www.dropbox.com/sh/v8gu9ic5ht1f6hj/AAACckYFTRZu-dfJG17ZaVgLa?dl=1) |
| enc-dec, PointNet++ | images (1 view) | object | [link](https://www.dropbox.com/sh/lotm1tas810oiip/AABluumM2UccoGcAsJzFsPOma?dl=1) |
| enc-dec, PointNet++ | images (3 view) | participant | [link](https://www.dropbox.com/sh/arp2ujgj15j0wuk/AACvKquP9-1zhd--199rpHuda?dl=1) |
| enc-dec, PointNet++ | images (1 view) | participant | [link](https://www.dropbox.com/sh/x2csef9nhnuw231/AAAZS7cWh7OFsBXbEuz4R_maa?dl=1) |

## Other Data

- [object model voxelizations]()
- Pre-computed "prediction data":
  - [simple-joints]()
  - [relative-joints]()
  - [skeleton]()
  - [mesh]()
  - [images]()
