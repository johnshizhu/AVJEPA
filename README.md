# AV-JEPA: Audio Video Joint Embedding Predictive Architecture

Audio Video Joint Embedding Predictive Architecture is an attempt at developing a models that can create versatile multimodal representations through self-supervised training. 
A write-up on our analysis can be found here: 

AV-JEPA: John Zhu, Boluo Ge

Original V-JEPA: Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran*, Nicolas Ballas*

## Method
AV-JEPA pretraining is purely self-supervised feature learning.


<img src="https://github.com/facebookresearch/jepa/assets/7530871/72df7ef0-2ef5-48bb-be46-27963db91f3d" width=40%>
&emsp;&emsp;&emsp;&emsp;&emsp;



## Visualizations


## Code Structure (same as in VJEPA)

**Config files:**
All experiment parameters are specified in config files (as opposed to command-line arguments). See the [configs/](configs/) directory for example config files. Note, before launching an experiment, you must update the paths in the config file to point to your own directories, indicating where to save the logs and checkpoints and where to find the training data.


```
.
├── app                       # the only place where training loops are allowed
│   ├── vjepa                 #   Video JEPA pre-training
│   ├── main_distributed.py   #   entrypoint for launching app on slurm cluster
│   └── main.py               #   entrypoint for launching app locally on your machine for debugging
├── evals                     # the only place where evaluation of 'apps' are allowed
│   ├── image_classification  #   training an attentive probe for image classification with frozen backbone
│   ├── video_classification  #   training an attentive probe for video classification with frozen backbone
│   ├── main_distributed.py   #   entrypoint for launching distributed evaluations on slurm cluster
│   └── main.py               #   entrypoint for launching evaluations locally on your machine for debugging
├── src                       # the package
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
└── configs                   # the only place where config files are allowed (specify experiment params for app/eval runs)
    ├── evals                 #   configs for launching vjepa frozen evaluations
    └── pretrain              #   configs for launching vjepa pretraining

```

## Data preparation (same as in VJEPA)

### Video Datasets
V-JEPA pretraining and evaluations work with many standard video formats.
To make a video dataset compatible with the V-JEPA codebase, you simply need to create a `.csv` file with the following format and then specify the path to this CSV file in your config.
```
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
...
```
Since V-JEPA is entirely unsupervised, the pretraining code will disregard the `$integer_class_label` in the CSV file.
Thus, feel free to put a random value in this column.
However, if you wish to run a supervised video classification evaluation on your video dataset, you must replace ```$integer_class_label``` with the ground truth label for each video.


## Launching AV-JEPA pretraining

### Local training
If you wish to debug your code or setup before launching a distributed training run, we provide the functionality to do so by running the pretraining script locally on a multi-GPU (or single-GPU) machine, however, reproducing our results requires launching distributed training.

The single-machine implementation starts from the [app/main.py](appmain.py), which parses the experiment config file and runs the pretraining locally on a multi-GPU (or single-GPU) machine.
For example, to run V-JEPA pretraining on GPUs "0", "1", and "2" on a local machine using the config [configs/pretrain/vitl16.yaml](configs/pretrain/vitl16.yaml), type the command:
```bash
python -m app.main \
  --fname configs/pretrain/vitl16.yaml \
  --devices cuda:0 cuda:1 cuda:2
```

### Distributed training
To launch a distributed training run, the implementation starts from [app/main_distributed.py](app/main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to launch a distributed pre-training experiment using the config [configs/pretrain/vitl16.yaml](configs/pretrain/vitl16.yaml), type the command:
```bash
python -m app.main_distributed \
  --fname configs/pretrain/vitl16.yaml \
  --folder $path_to_save_stderr_and_stdout \
  --partition $slurm_partition
```

## Launching Evaluations

### Local training
If you wish to debug your eval code or setup before launching a distributed training run, we provide the functionality to do so by running the evaluation script locally on a multi-GPU (or single-GPU) machine, however, reproducing the full eval would require launching distributed training.
The single-machine implementation starts from the [eval/main.py](eval/main.py), which parses the experiment config file and runs the eval locally on a multi-GPU (or single-GPU) machine.

For example, to run ImageNet image classification on GPUs "0", "1", and "2" on a local machine using the config [configs/eval/vitl16_in1k.yaml](configs/eval/vitl16_in1k.yaml), type the command:
```bash
python -m evals.main \
  --fname configs/eval/vitl16_in1k.yaml \
  --devices cuda:0 cuda:1 cuda:2
```


### Distributed training
To launch a distributed evaluation run, the implementation starts from [eval/main_distributed.py](eval/main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to launch a distributed ImageNet image classification experiment using the config [configs/eval/vitl16_in1k.yaml](configs/eval/vitl16_in1k.yaml), type the command:
```bash
python -m evals.main_distributed \
  --fname configs/eval/vitl16_in1k.yaml \
  --folder $path_to_save_stderr_and_stdout \
  --partition $slurm_partition
```

Similarly, to launch a distributed K400 video classification experiment using the config [configs/eval/vitl16_k400.yaml](configs/eval/vitl16_k400.yaml), type the command:
```bash
python -m evals.main_distributed \
  --fname configs/eval/vitl16_k400.yaml \
  --folder $path_to_save_stderr_and_stdout \
  --partition $slurm_partition
```

---

### Setup

Run:
```bash
conda create -n jepa python=3.9 pip
conda activate jepa
python setup.py install
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## VJEPA Citation
Please review [VJEPA](https://github.com/facebookresearch/jepa)
```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael, and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv:2404.08471},
  year={2024}
}
