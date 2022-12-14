#Capsule Neural Networks
<h5>This repository contains the implementation of the Capusle Neural Network (https://arxiv.org/abs/1710.09829) </h5>


# Instructions


```bash
  Clone the Repo
  pip install requirements.txt
  #  Follow the article to save the .kaggle token file  or export the authentication api token
  #  https://github.com/Kaggle/kaggle-api#api-credentials
  
  cd Code
  python3 -u test_capsnet.py.py -ImageDownload
  #  -ImageDownload arg to get the data files
  #  -Train arg to enter train Phase
  
  python3 -u Baseline_CNN.py
  #  -ImageDownload arg to get the data files

  python3 -u explain.py
  #  Run the test_capsnet.py or Baseline_CNN.py with -ImageDownload arg to get the data before running the explain.py file
  
```


# Dataset
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

