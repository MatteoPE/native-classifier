# NativeClassifier

## Requirements

- Create a virtual environment

```bash
virtualenv env
```

- Install all the requirements in the virtual environment

```bash
source env/bin/activate
pip3 install -r requirements.txt
```

# Train the model

- Run the script that trains the model

```bash
source env/bin/activate
python train_model.py
```

# Load the model inside a Python script

- Add these lines to your Python script

```python
from keras.models import load_model

model = load_model('model.h5')
model.summary()
```