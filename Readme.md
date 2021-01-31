# Document modeling with Generative Adversarial Networks

### [A] Requirements (One-time setup)
1. Requires Python 3 (tested with `3.6.1`)
2. Create a virtual environment to run the model, using below commands:
        a) $ conda create -y --name dmgan python=3.6
	b) $ conda activate dmgan
        c) $ conda install pip 
3. Install the remaining dependencies and libraries, using below commands:
        a) $ pip install -r requirements.txt
        b) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/python -c "import nltk; nltk.download('punkt')"
        c) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install nltk --upgrade
        d) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install numpy --upgrade
        e) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install scipy --upgrade
        f) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install sklearn --upgrade
        g) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install scikit-learn --upgrade
        h) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install pandas --upgrade
        i) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install tensorflow==1.4.0
        j) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install tensorboard==1.0.0a6
        k) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install matplotlib
        l) $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/pip install seaborn


### [B] Prepare the raw input dataset (consisting of 18,846 documents), and split over train-test-validation datasets
        $ python prepare.py
        # $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/python prepare.py 


### [C] Pre-process the input raw data to the vectorized format expected by the model 
        $ python preprocess.py --input <path to input dataset> --output <path to preprocessed output dataset> --vocab <path to vocab file>
        # $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/python preprocess.py --input data --output preprocessed_data --vocab data/20newsgroups.vocab 
Here:
- 1st column is label and 2nd column is document body in the CSVs
- -> `training.csv` (13,192 documents); `validation.csv` (1,884 documents); `test.csv` (3,769 documents); 
- Vocabulary file used is placed in [`data/20newsgroups.vocab`]


### [D] Training of model
        $ python train.py --dataset <path to preprocessed dataset> --model <path to model output directory>
        # $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/python train.py --dataset preprocessed_data --model results --num_steps 2500 --log_every 100 --save_every 200 

- To view Tensorboard graphs, plots, etc. run this in new terminal:
        $ tensorboard --logdir results/logs/

- To view additional parameters: 
        $ python train.py --help

### [E] Extracting document vectors 
        $ python vectors.py --dataset <path to preprocessed dataset> --model <path to trained model directory>
        # $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/python vectors.py --dataset preprocessed_data --model results 


### [F] Evaluating results
        $ python evaluate.py --dataset <path to preprocessed dataset> --model <path to trained model directory>
        # $ /Users/harshirao/opt/anaconda3/envs/dmgan/bin/python evaluate.py --dataset preprocessed_data --model results 

