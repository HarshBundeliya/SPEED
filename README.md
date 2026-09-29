# **SPEED: Selective Prediction for Early Exit DNNs**

This repository contains the official implementation of the paper **SPEED: Selective Prediction for Early Exit Deep Neural Networks (DNNs)**.

## **Requirements**

This implementation is built using the **Hugging Face Transformers** library.

## **Training**

To fine-tune a pre-trained language model and train the DC classifiers, run the following command:

```bash
python3 main.py --pretrain --dctrain --src SST2 --tgt imdb
```

## **Code Acknowledgement**
We acknowledge the [DAdEE](https://github.com/Div290/DAdEE) repository and thank them for making their source code publicly available, which was instrumental in our work
