<p align="center"><img src="images/banner.png"/></p>

<p align="center"><b>Game Ideas Generation Using Neural Networks</b></p>
<br>

# 🔧 Requisites

- Pyhton 3.8
- Numpy 1.2
- TensorFlow 2.8
- Keras 2.8

# 🚀 Usage

Before you can generate ideas for games you must train the neural network, for that run:

```
python MLGamedevIdeas.py -train
```

This will train the neural network with all the descriptions in the file '_final_data_new.json.gz_' (about 80000), with 20 epochs.
Depending on your hardware this may take quite a while.

To make the training process less time consuming you can use fewer game descriptions:

```
python MLGamedevIdeas.py -games 100 -train
```

You can also use less epochs:

```
python MLGamedevIdeas.py -epochs 10 -train
```

For good results I recommend values of '_loss_' below 2. Consult the parameters to better adjust the training:

```
python MLGamedevIdeas.py -h
```

Once the neural network has been trained, a file '_weights.hf5_' will have been generated.

> Note that you will have to retrain the neural network if you change the parameters.

Now you can start generating crazy ideas:

```
python MLGamedevIdeas.py
```

## 📜 License

Code released under [MIT License](https://github.com/FronkonGames/Machine-Learning-Game-Ideas/blob/main/LICENSE.md).

