### [Multimodal Video Similarity Challenge](https://algo.browser.qq.com/)
#### [@CIKM 2021](https://www.cikm2021.org/analyticup) 
This is the official tensorflow version baseline

#### 1. data
Please download the 'data' directory and verify the integrity first, then prepare 'data' according to [data/README.md](data/README.md)

#### 2. code description
- [config.py](config.py) contains all the configuration
- [data_helper.py](data_helper.py) handles data processing and parsing
- [evaluate.py](evaluate.py) evaluates the model
- [inference.py](inference.py) generate the submitting file
- [metrics.py](metrics.py) outputs the metric when training
- [model.py](model.py) builds the model
- [train.py](train.py) is the training entry

#### 3. Install the dependency
```bash
pip install -r requirements.txt
```

#### 4. Train
```bash
python train.py
```

#### 5. Inference
```bash
python inference.py
```

#### 6. Evaluate
```bash
python evaluate.py
```

