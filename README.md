# Machine Learning
A playground, collections of machine learning patterns I'm trying

## Use ipykernels
1. Create new venv and activate
    ```
    python -m venv {kernel_name}
    .\{kernel_name}\Scripts\activate
    ```
2. Install ipykernel
    ```
    pip install ipykernel
    ```
3. Activate ipykernel in jupyter lab
   ```
   python -m ipykernel install --name={kernel_name}
   ```

## Tensorflow
For tensorflow that I'm using, I use `!pip install "tensorflow<2.11"` to install tensorflow version 2.10 because it supports local GPU acceleration for faster training.

```python
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
>Use different environment for each directory. Some will not be compatible with other (*tensorflow*, *mediapipe*, *pytorch*)