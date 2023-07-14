# Reimplementation of the Forward-Forward Algorithm

Our Paper: [https://arxiv.org/abs/2307.04205](https://arxiv.org/abs/2307.04205)

This is a reimplementation of Geoffrey Hinton's Forward-Forward Algorithm in Python/Pytorch. Majority of the code in the folder ```official_python_implementation``` is taken from [here](https://github.com/loeweX/Forward-Forward).

&rarr; [Original Paper](https://arxiv.org/abs/2212.13345)


This code covers the experiments described in section 3.3 ("A simple supervised example of FF") of the paper and 
achieves roughly the same performance as the official Matlab implementation (see Results section).

In addition to a re-implementation of section 3.3 ("A simple supervised example of FF") of the paper, we have extended the forward-forward algorithm to a sentiment analysis task. We have also performed explorations with thresholds and different activation functions. Our findings have been summarized here: [Report](https://drive.google.com/file/d/15m5rq16Z0IG8nOqyxyRreH0Fk7oeO7kN/view?usp=share_link) and [Video](https://www.youtube.com/watch?v=hl6uD0mXMAw&t=1s&ab_channel=JonahKornberg)

## How to Use

- Install required dependencies from requirements.txt
- update config.py with the required parameters for choosing a dataset (MNIST/CIFAR10/Sentiment) and model architecture
- From the official_python_implementation folder, run the following command:
```bash
python main.py
```
