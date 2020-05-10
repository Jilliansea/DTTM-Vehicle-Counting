# DTTM-Vehicle-Counting
## Introduction
In this repo, we include the submission to AICity Challenge 2020 Vehicle Counts by Class at Multiple Intersections (Didi Chuxingsubmission).

We propose a robust and fast vehicle turn-counts at intersections via an integrated solution from detection, tracking and trajectory modeling. 
Our team ranks 6th in Public leaderboard and models of our algorithms are not trained with any extra datasets. 

## Installation
Our code is tested on Tesla P40, 24G with following setting:
(1) Linux
(2) Python 3.6
(3) PyTorch 1.1 or higher
(4) CUDA 10
(5) NCCL 2
(6) GCC 4.9 or higher

The fast way to install our code is running commond as follows:
```
pip3 install -r requirements.txt
```
## Test
### Get videoes directory of Track1:
After downloading packages of AICity Challenge 2020 Track1, please unzip and **$DirPath\_to\_Track1\_AIC20\_track1** is the final directory after unzip.
### Download detection model:
Download our detection model on  
Then our code can be run as follows [RetinaNetNas-FPN](https://drive.google.com/drive/folders/1cEBRVSXJH_f6BNr_LvISRZmuMuIXnXPC)
### Run test code 
```
python  multi_process.py --video_dir=$DirPath_to_Track1_AIC20_track1
```
The final counting results will be stored in **count_nums/**  

## Reference
[Mmdetection](https://github.com/open-mmlab/mmdetection)

 
