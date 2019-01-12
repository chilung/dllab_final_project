# CINIC10影像分類執行方法
1. 資料準備：執行A或B
- A.下載封裝好的cinic10_train.npz以及cinic10_test.npz，並將封裝檔放在cinic10資料夾內
-   https://drive.google.com/open?id=1Mqdc3HWEZMkQqnwPA70tbwH95KncrCEr
-   https://drive.google.com/open?id=1BSnrVXoL9IOdWyjfrKQlyBsoSadghXxx
- B.自行產生NPZ file，方法如下：
```python
import os
import sys
module_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(module_path)
import distiller
import apputils
from apputils import CINIC10_NPZ

cinic10_npz = CINIC10_NPZ()
cinic10_npz.save_npz()
```
2. 執行影像分類
- $ python cinic_classification.py
- 會自動載入cinic10資料夾內的checkpoint.pth.tar作為使用model，以及步驟1中製作的測試資料集(cinic10/cinic10_test.npz)做inference
- 執行結果如下所示：
```
Start update
cpu_metric: 84.63888888888889, cpu: 7.969727993011475 s,gpu_metric: 84.63888888888889, gpu:7.921760082244873, num of weight:274042.0, size: 1.0453872680664062 MB
```

# Super Resolution 執行方法
```
1. Environment preparation:
===========================
cd /home/dllab/ngraph
source ~/ngraph/onnx/bin/activate
pip install --upgrade pip
pip3 install --force-reinstall torch==0.4.0
cd ~/TensorRT-5.0.2.6/python
pip3 install tensorrt-5.0.2.6-py2.py3-none-any.whl
pip install matplotlib
pip install imageio
pip install tqdm
pip install scikit-image

2. NOTICE:
==========
Make sure your testset are put in directory:
    [TESTDATADIR]/DIV2K/DIV2K_valid_HR
          and
    [TESTDATADIR]/DIV2K/DIV2K_valid_LR_bicubic

3. Usage:
=========
in directoy ./super_resolution/EDSR-PyTorch/src
$ TESTDATADIR=[PATH to DIV2K VALIDATE DATA] python3 example.py
```

# 物件偵測執行方法
- $ python clothes_recognize.py
- 會自動擷取 clothes/cfg/valid.txt 內容所列檔案名稱進行物件偵測
- 此檔案內容目前為 /tmp/dataset-nctu/clothes/clothes_test/images/2096.jpg ...
- 或是
- $ TESTDATADIR="/tmp/dataset-nctu/clothes/test/" python3 clothes_recognize.py
- 指定檔案位置。程式將先列出/tmp/dataset-nctu/clothes/test/images/ 下所有jpg檔案，
- 並更新 clothes/cfg/valid.txt 後進行偵測
- 執行前，請先下載weights，並置於目錄 clothes/cfg 之下。weights位於
- https://drive.google.com/open?id=17TPyHT1Fo-4bQcCgSfwtqgbXF2AZ6Y8d

# 評分環境
- 機器: NCHC Aitrain container
- Python environment as LAB5
# 使用方法
- Download this repository to benchmark your project
```shell
$ git clone https://github.com/nctu-arch/NCTU_DLSR_final_project.git
```
- Preparation: install requirements
```shell
$ cd NCTU_DLSR_final_project
$ pip3 install -r requirements.txt
```
- import benchmark
```python
from benchmark import benchmarking
```
- Benchmarking function usage - `benchmarking(team, task, model, preprocess_fn, *pre_args, **pre_kwargs)`
  - team: 
      - 1~12
  - task: 
    - 0: classification
    - 1: super resolution
    - 2: objection detection
  - model: pytorch 
    - 目前計算 pytorch model weight 數量及大小
  - preprocess_fn, *pre_args, **pre_kwargs: 
    - 前處理 function， 可以轉換 data format
    - 參數 可自定義, 無則 None
- 撰寫 Inference code
```python
net = resnet18() # define model 
@benchmarking(team=12, task=0, model=net, preprocess_fn=None)
def inference_fn(*args, **kwargs):
    dev = kwargs['device']
    if dev == 'cpu':
        metric = do_cpu_inference()
        ...
    elif dev == 'cuda':
        metric = do_gpu_inference()
        ...
    return metric
```
# Test Categories
* CINIC-10
    * Baseline
        * CINIC-10 test data
    * Accuracy Ranking
        * private test data
    * Model size
    * CPU inference time
    * GPU inference time
* DIV2K
    * Baseline
        * DIV2K x2 validtion data
    * PSNR Ranking
        * private test data
    * Model size
    * CPU inference time
    * GPU inference time
* Clothes
    * Baseline: 
        * Validation data
    * F-score Ranking
        * ITRI test data
    * Model size
    * CPU inference time
    * GPU inference time
# What to Submit?
* Any source code you used in your project.
* Create a team directory named teamX including 'Classification','Object Detection' and 'Super Resolution' to push each task respectively.
```
e.g..
.
├── team11
└── team12
    ├── Classification
    ├── Object Detection
    └── Super Resolution
```
# How to Submit?
* As a student, you can apply for a GitHub Student Developer Pack, which offers unlimited private repositories.
* Fork this repository, and then make your forked repo duplicated. (Settings -> Danger Zone)
* Add nctu-arch as collaborator. (Settings -> collaborator)
* After deadline we will pull your source code for open review.
* Please describe the external plugins you used and its usage precisely.
# Example usage
  ```
  TESTDATADIR="/tmp/dataset-nctu/clothes/clothes_test/" python3 clothes_recognize.py
  ```
  * default: overide team 12 data
# Score sheet link
http://140.113.213.76/
