# Directory 
## Root
The `${ROOT}` is described as below.  
```  
${ROOT} 
|-- common 
|-- data  
|-- demo
|-- main  
|-- output  
```  
* `common` contains kernel codes for ClothWild.  
* `data` contains required data and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result.  


## Required data
You need to follow directory structure of the `data` as below. 
```
${ROOT} 
|-- data  
|   |-- base_data
|   |   |-- human_models
|   |   |   |-- SMPL_FEMALE.pkl
|   |   |   |-- SMPL_MALE.pkl
|   |   |   |-- SMPL_NEUTRAL.pkl
|   |   |-- smplicit
|   |   |   |-- checkpoints
|   |   |   |   |-- hair.pth
|   |   |   |   |-- pants.pth
|   |   |   |   |-- shoes.pth
|   |   |   |   |-- skirts.pth
|   |   |   |   |-- upperclothes.pth
|   |   |   |-- clusters
|   |   |   |   |-- clusters_hairs.npy
|   |   |   |   |-- clusters_lowerbody.npy
|   |   |   |   |-- clusters_shoes.npy
|   |   |   |   |-- indexs_clusters_tshirt_smpl.npy
|   |-- preprocessed_data
|   |   |-- densepose
|   |   |-- gender
|   |   |-- parse
|   |   |-- smpl_param
|   |-- ...
```
* `base_data/human_model_files` contains `smpl` 3D model files. Download the files from [[smpl](https://smpl.is.tue.mpg.de/)].
* `base_data/smplicit` contains 3D cloth generative model (`SMPLicit`) files. Download the files from [[smplicit](https://github.com/enriccorona/SMPLicit)].
* `preprocessed_data` is required for training and testing stages. Download it from [[preprocessed_data](https://drive.google.com/file/d/1N6hPKMDgk1SRoKJDx5cGgoSlZEhKTT-4/view?usp=sharing)].


## Dataset
```  
${ROOT} 
|-- data  
|   |-- ...
|   |-- DeepFashion2
|   |   |-- data
|   |   |   |-- train
|   |   |   |-- DeepFashion2_train.json
|   |-- MSCOCO
|   |   |-- images
|   |   |   |-- train2017
|   |   |   |-- val2017
|   |   |-- parses
|   |   |-- annotations
|   |   |   |-- coco_wholebody_train_v1.0.json
|   |   |   |-- coco_wholebody_val_v1.0.json
|   |   |   |-- coco_dp_train.json
|   |   |   |-- coco_dp_val.json
|   |-- PW3D
|   |   |-- data
|   |   |   |-- imageFiles
|   |   |   |-- sequenceFiles
|   |   |   |-- 3DPW_test.json
```  
* Download DeepFashion2 parsed data [[data](https://github.com/switchablenorms/DeepFashion2)] [[annot](https://drive.google.com/drive/folders/1HKv_wt3F23Rjnig7TwaOxHWTl5sGrdbC?usp=sharing)]
* Download MSCOCO data, parses (LIP dataset), and densepose [[data](https://github.com/jin-s13/COCO-WholeBody)] [[parses](https://drive.google.com/file/d/1I7-p0w6GSGOryBkFA348CeHSmcJS6ro4/view?usp=sharing)] [[densepose](https://drive.google.com/drive/folders/1HKv_wt3F23Rjnig7TwaOxHWTl5sGrdbC?usp=sharing)]
* Download 3DPW parsed data [[data](https://virtualhumans.mpi-inf.mpg.de/3DPW/)] [[annot]](https://drive.google.com/drive/folders/1HKv_wt3F23Rjnig7TwaOxHWTl5sGrdbC?usp=sharing)
* All annotation files follow [MSCOCO format](http://cocodataset.org/#format-data). If you want to add your own dataset, you have to convert it to [MSCOCO format](http://cocodataset.org/#format-data).  


### Output
```  
${ROOT}  
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```  
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.  
* `log` folder contains training log file.  
* `model_dump` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files generated in the testing stage.  
* `vis` folder contains visualized results.  