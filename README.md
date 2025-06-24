# Installation

* install all dependence via pip

    ```shell
    pip3 install -r requirements.txt
    ```


# Dataset
The link to the data set we generated based on the original text is as follows:
* Nyu Data: link
* RealSR: link
* DIV2K: link
* REDS: link
* MANGA109 : link


# Experiment

## Generate Prompt Identities

Use the `generate.py' python file to generate task identifiers pool. Or use the pool we have generated, the numpy file in the link:

Put them in the same directory as `main_train_continue.py'. Now, your directory structure should look like this:

```tree
|  |--data
|  |--models
|  |--utils
|  |--options
|--main_train_continue.py
|--main_test_continue.py
|--inference.py
|--generate.py
|--clip_feature_lr_{dataset_name1}_{degradation1}.npy
|-- ......
|--clip_feature_lr_{dataset_name4}_{degradation4}.npy
```



## Train Our Method

```shell
python3 main_train_continue --opt options/swinir/continue/x4/freezeall/train_continue_freezeall_promptv6_1_2_3_ps[3]_pl[12]_bswinir500_p012345_fmsr_clip_d180_s4_t32_nrdem_brcmj.json
```

Please change the `gpu ids' and all data paths in the configuration file before training.

For example:

```json
{ 
	......
	"gpu_ids":[0],
	......
	"dataroot_H":[
    	path1,
    	path2,
        path3,
        path4
    ]
}
```

# Test

Use the command:

```shell
python3 main_test_continue --opt options/{opt_json_path} -d cuda:0 
```

This program will automatically test all files in the path included in test in the configuration file.
