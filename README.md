# Introduction 
This module is responsible for converting bolf to coco format and evaluating the converted annotation on cocometrics.

# Getting Started
1.  Installation process
First of all create conda environment and install cocometrics
    ```
    conda create -n torch-cpu Python=3.8
    conda activate torch-cpu
    conda install pytorch torchvision cpuonly -c pytorch
    pip install -U openmim
    mim install mmengine
    mim install "mmcv-lite>=2.0.0rc1"
    git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
    cd mmdetection3d
    pip install -v -e .
    mim install "mmdet3d>=1.1.0"
    ```

2. Command for running the cocometrics
* For running bolf to coco converter
    ```
    conda activate torch-cpu
    python -m cocometrics.convertor_main -i /path/to/bolf-json/dir/ -o /path/to/output-dir/
    ```
    run this commmand for gt and pred one by one

* for running coco metrics 
    ```
    conda activate torch-cpu
    python ./coco_evaluate_main.py -g /path/to/ground-truth-json/dir/ -p /path/to/prediction-json/dir/
    ```

argument details for both the files

```
python convertor_main.py --help
usage: convertor_main.py [-h] -i INPUT_DIR -o OUTPUT_DIR

Process input directory and specify output path.        

optional arguments:
  -h, --help            show this help message and exit 
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Input directory path
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory path
```

```
python coco_evaluate_main.py --help

usage: coco_evaluate_main.py [-h] -g GT_PATH -p PRED_PATH

Process input directory and specify output path.

optional arguments:
  -h, --help            show this help message and exit
  -g GT_PATH, --gt-path GT_PATH
                        Ground truth annontation directory path
  -p PRED_PATH, --pred-path PRED_PATH
                        Prediction annontation directory path

```

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
[Lavpreet Singh](Lavpreet.singh@in.bosch.com)
