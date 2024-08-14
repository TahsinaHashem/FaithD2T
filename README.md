# Dataset and Code for Generating Faithful and Salient Text from Multimodal Data (INLG 2024) paper.

## Dataset Download Link
Please download the Real-estate House dataset from [here](https://drive.google.com/file/d/16h4EQlgs-x4hypu4vAq2tuL0EzvrCecO/view?usp=drive_link).

## Training & Evaluation
The scripts for training and evaluation are in the main directory.

**Training**
```bash
#Finetune blip2-flan-t5-xl model and save checkpoint
python trainPart1.py
python trainPart2.py
```

**Evaluation**
```bash
#Perform inference on test data
python evalPart1.py
python evalPart2.py
```

## Cite
If you find this work useful for your research, please consider citing.
<pre><tt>@inproceedings{hashem2024generating,
  author    = "Hashem, Tahsina and Wang, Weiqing and Wijaya, Derry Tanti and Ali, Mohammed Eunus and Li, Yuan-Fang"
  title     = "Generating Faithful and Salient Text from Multimodal Data",
  booktitle = "INLG",
  year      = "2024",
}</tt></pre>

## Acknowledgements
This implementation is based on the code provided by [huggingface/peft](https://github.com/huggingface/peft/blob/main/examples/int8_training/fine_tune_blip2_int8.py)
