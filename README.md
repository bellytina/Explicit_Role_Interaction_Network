# Explicit_Role_Interaction_Network
Here is the code for "Explicit Role Interaction Network for Event Argument Extraction".

## Data preprocessing
1.Split datasets and preprocess data from [ACE2005 preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing)

2.Process the data into the format in [./data/ace/example.json](https://github.com/bellytina/Explicit_Role_Interaction_Network/blob/8ef282edeab65a8fa5d269acadab16b61c67251c/data/ace/sample.json). I add the *flag* id to indicate whether this event type is accurately classified in the upstream task

## Event Detection 
In this paper, we employ a pre-trained BERT model and stack a softmax layer.

Replace the golden start and end index of triggers with ED predicted results in both files dev.json and test.json.

## Event Argument Extracion
python train.py
