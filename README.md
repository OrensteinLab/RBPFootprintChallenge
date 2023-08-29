## Requirements
Python 3.9.15
TensorFlow version: 2.11.0
Pandas version: 1.5.2
NumPy version: 1.24.1

## Installation

Install packages and clone repository :

```bash
pip install -r requirements.txt

cd RBPFootprintChallenge
git clone https://github.com/OrensteinLab/RBPFootprintChallenge.git
tar zxvf ./Data/Data.gz
tar zxvf ./Data/gencode.v39.transcripts.fa.gz
```

## Scripts

### Predict SHAPE to transcriptome
```
python predSHAPEtoTranscriptome.py
```

###  Train model (based on DLPRB)
```
python DLPRB_Model_Train.py
```

### Predict protein binding scores
```
python PredPB.py
```

### Find max predicted protein binding scores
```
python findMax.py
```

