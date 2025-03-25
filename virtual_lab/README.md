# About the Virtual Lab

> This file includes the training and inference of the predictable model for getting `g-factor`, a property value of nano helix material. 
> This file is used to simulate the process of doing experiments with specific software in a high efficient way.

# How to run?

### Step 1
If you want to re-train the model:

```shell
cd src
python training.py
```
The model will be trained and save the weights automatically. 


### Step 2

After training (as default), you can run the server by:

```shell
cd src
python inference.py 
```

This will run the flask app to make it listen to a specific port.

### Step 3

Test the API:

```shell
python test_tool.py
```
