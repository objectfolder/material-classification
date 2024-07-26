# Material Classification

All objects are labeled by seven material types: ceramic, glass, wood, plastic, iron, polycarbonate, and steel. The task is formulated as a single-label classification problem. Given an RGB image, an impact sound, a tactile image, or their combination, the model must predict the correct material label for the target object.

## Usage

#### Data Preparation

The dataset used to train the baseline models can be downloaded from [here](https://www.dropbox.com/scl/fo/ymd3693807jucdxj7cj1k/AMtyNZgmC1ynxFWZtVsV5gI?rlkey=hr1y85tzadepw7zb5wb9ebs0b&st=xr2keno9&dl=0)

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
# Train FENet as an example
python main.py --model FENet --config_location ./configs/FENet.yml \
               --modality_list vision touch audio --batch_size 256 \
               --lr 1e-3 --weight_decay 1e-2 --exp FENet_vision_touch_audio
```

Evaluate the best model in *FENet_vision_touch_audio*:

```sh
# Evaluate FENet as an example
python main.py --model FENet --config_location ./configs/FENet.yml \
               --modality_list vision touch audio --batch_size 256 \
               --lr 1e-3 --weight_decay 1e-2 --exp FENet_vision_touch_audio \
               --eval
```

#### Add your own model

To train and test your new model on ObjectFolder Cross-Sensory Retrieval Benchmark, you only need to modify several files in *models*, you may follow these simple steps.

1. Create new model directory

   ```sh
   mkdir models/my_model
   ```

2. Design new model

   ```sh
   cd models/my_model
   touch my_model.py
   ```

3. Build the new model and its optimizer

   Add the following code into *models/build.py*:

   ```python
   elif args.model == 'my_model':
       from my_model import my_model
       model = my_model.my_model(args)
       optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   ```

4. Add the new model into the pipeline

   Once the new model is built, it can be trained and evaluated similarly:

   ```sh
   python main.py --model my_model --config_location ./configs/my_model.yml \
                  --modality_list vision touch audio --batch_size 256 \
                  --lr 1e-3 --weight_decay 1e-2 --exp my_model_vision_touch_audio
   ```

## Results on ObjectFolder Material Classification Benchmark

The 1, 000 objects are randomly split into train/validation/test = 800/100/100, and the model needs to generalize to new objects during the testing process. Furthermore, we also conduct a cross-object experiment on ObjectFolder Real to test the Sim2Real transferring ability of the models, in which the 100 real objects are randomly split into train/validation/test = 60/20/20.

#### Results on ObjectFolder

<table>
    <tr>
        <td>Method</td>
        <td>Vision</td>
        <td>Touch</td>
        <td>Audio</td>
        <td>Fusion</td>
    </tr>
    <tr>
        <td>ResNet</td>
        <td>91.89</td>
        <td>74.36</td>
      	<td>94.91</td>
        <td>96.28</td>
    </tr>
  	<tr>
        <td>FENet</td>
        <td>92.25</td>
        <td>75.89</td>
      	<td>95.80</td>
        <td>96.60</td>
    </tr>
</table>


#### Results on ObjectFolder Real

<table>
    <tr>
        <td>Method</td>
        <td>Accuracy</td>
    </tr>
    <tr>
        <td>ResNet w/o pretrain</td>
        <td>45.25</td>
    </tr>
  	<tr>
        <td>ResNet</td>
        <td>51.02</td>
    </tr>
</table>
