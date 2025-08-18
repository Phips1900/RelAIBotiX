# Build Steps:

* Create python>=3.10.16 virtual environment
* Install requirements: pip install -r requirements.txt

**All commands must be executed from Project Root.** 

Let us say, we want to train Franka dataset. Currently, we have 5 classes: idle, move, pick, carry, place.

Dataset
-------
* Copy a dataset to train (franka.h5) in directory: ```<proj_root>/dataset/training```.
* Set dataset path in respective yaml -> ``configs/data/franka.yaml``. (See yaml file).

Training
--------
* See ``configs/training/franka.yaml`` to get an understanding of how training configuration is customised. 
  * You can change training hyperparameters here. By default, epoch size is 50.  
  * Update the checkpoint filename as per wish.
* Additionally, review ``configs/model/cnn_transformer.yaml``. Model architecture related hyperparameters can be set here.
* Run Training: ```python src/training_pipeline.py model=cnn_transformer robot=franka```
* Training logs are saved in ``/outputs/franka/training/\<timestamp>``. 
* Tensorboard logs are saved version-wise in ``tb_logs/``. With every training execution, the version increments for respective category.
  * Run command: ```tensorboard --logdir tb_logs/```. Go the port address to view the results. 
* Trained models are saved in ```models/franka/ckpts/```. 
* Set the ckpt path in ``configs/model_selector/rule_based.yaml`` to 'checkpoint' under 'franka' structure. This will our pre-trained model selection.

Inference
---------
* Copy an unseen dataset in ```<proj_root>/dataset/inference```.
* Either do this:
  * Set ``data_path`` and ``robot`` in ```configs/inference_config.yaml```.
* Or:
  * Run Inference by passing dataset path and robot type as argument -
    * ```python src/inference_engine.py data_path="\<path>" robot=ur5``` or franka (automatically maps model through model_selector). 
* Inference metric logs are saved in ```outputs/franka/inference/\<timestamp>```.

* If you want to execute programs through PyCharm IDE (for live-debugging),  
  * Go to Run > Edit Configurations 
  * Set "Working directory" to: <proj_root> for Current File.
  * Current working directory for all execution environments MUST BE <proj_root>, otherwise debugging will fail.

Please ignore following section. Todo

## Experiments 
pick and place trials: move->pick->carry->place 
### sample 1
* w/o shuffle
* 20 trials
* window size: 100
* epoch 48 best model.
* accuracy: 99.6%

### sample 2
* w/o shuffle
* 20 trials
* window size: 200
* epoch 31: 99.6%. best model 46
* accuracy: 99.7%

### sample 3
* w/o shuffle
* 20 trials
* window size: 300
* best model epoch 47
* accuracy: 99.7%

### sample 4
* with shuffle
* 20 trials
* window size: 300
* best model epoch 
* accuracy: 97.7 until epoch 7
* no difference in performance

## Key Points:
For sequence labeling:
Each window of size w contributes w label predictions.
With stride = 1, your number of windows per trial = (T - w + 1), where T = number of timesteps per trial.
Total predictions = number of windows × window size
= (T - w + 1) × w
Let’s plug in some example numbers to illustrate the trend (assuming T = 1000 timesteps per trial):

Window Size (w)	Windows per trial (1000 - w + 1)	Predictions per trial = windows × w
100	            901	                                90,100
200	            801	                                160,200
300	            701	                                210,300

Now multiply that by 20 trials:
Sample 1: 90,100 × 20 = 1,802,000 predictions
Sample 2: 160,200 × 20 = 3,204,000 predictions
Sample 3: 210,300 × 20 = 4,206,000 predictions
Even if only a portion of those predictions land in a single cell of the confusion matrix (e.g., true positive for one class), the total raw counts scale up with window size. That’s why:

Confusion matrix scale in:
Sample 1 peaks at ~175k
Sample 2 reaches ~350k
Sample 3 climbs to ~500k

