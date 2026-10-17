# Code Origin Detector
Final project for the course Machine learning for statistical NLP: Advanced LT2326/LT2926


## Main repository tree structure
```text
.
├── data/               # contains all data
│   ├── processed           # preprocessed dataset
│   ├── raw                 # original dataset
├── docs
│   ├── experiments     # experiment details
│   ├── notebooks       # general notes
│   ├── reports         # submission content
├── run                 # run-all pipeline related content
├── src                 # code
```

## Init
### RUN ALL
To run everything, simply execute this command from the repository root.
```bash
bash run/RUN.sh
```
The file [run.conf](./run/run.conf) controls what [RUN.sh](./run/RUN.sh) executes and how. Outputs are logged in [run.log](./run/run.log).