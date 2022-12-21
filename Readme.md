# Document modeling with Generative Adversarial Networks
[Inspired from https://github.com/AYLIEN/adversarial-document-model]

The below steps need to be run to generate documents using this GAN model.

### Requirements
1. Ensure that Python 3 is installed before installing remaining dependencies 
2. Create a virtual environment (conda or pip3) to run the solution
- Create a `pip3` virtual environment to run the model, using below commands:

	`$ python3 -m venv dmgan`
	
	`$ source dmgan/bin/activate`                  
- Create a `conda` virtual environment to run the model, using below commands:

	`$ conda create -y --name dmgan python=3.6`
	
	`$ conda activate dmgan`
	
	`$ conda install pip`
3. Install the remaining dependencies and libraries, by running below command:   
        
	`$ pip install -r requirements.txt`


### Data Population
1. Run below command to prepare the raw input dataset (consisting of 18,846 documents), and split over train-test-validation datasets:  
        
	`$ python prepare.py` 
2. 3 new files (`training.csv` (13,192 documents); `validation.csv` (1,884 documents); `test.csv` (3,769 documents)) are populated in the `/data` folder. In each CSV file, the 1st column is the label and 2nd column is the raw text document body.


### Data Preprocessing
1. Run below command to pre-process the input raw data to the vectorized format expected by the model:  
        
	`$ python preprocess.py --input data --output preprocessed_data --vocab data/20newsgroups.vocab`

where: `input` is `path to input dataset`; `output` is `path to preprocessed output dataset`; `vocab` is `path to vocab file` 

2. 4 new files (`training.csv`; `validation.csv`; `test.csv`; `labels.txt`) are populated in the `/preprocessed_data` folder. In each CSV file, the 1st column is the label and 2nd column is the vectorized document body. The text file consists of the 20 groups of 20NewsGroups corpus.


### Model Training
1. Run below command to train the GAN model:
	
	`$ python train.py --dataset preprocessed_data --model results`
 
where: `dataset` is `path to preprocessed dataset`; `model` is `path to model output directory`

2. To view Tensorboard graphs, plots, etc., run below command in new terminal and open the generated URL link:
        
	`$ tensorboard --logdir results/logs/`

where: `logdir` is `path to results logs directory`

3. To view additional parameters: 
        
	`$ python train.py --help`


### Evaluating results
1. Run below command to evaluate the retrieval results: 
        
	`$ python evaluate.py --dataset preprocessed_data --model results` 
 
where: `dataset` is `path to preprocessed dataset`; `model` is `path to trained model directory`       
        
### Extracting document vectors 
1. Run below command to extract document vectors which will be saved in NumPy text format to the model directory: 
        
	`$ python vectors.py --dataset preprocessed_data --model results` 
 
where: `dataset` is `path to preprocessed dataset`; `model` is `path to trained model directory`

