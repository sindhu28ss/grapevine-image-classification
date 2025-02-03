#!/usr/bin/env python
# coding: utf-8

# In[105]:


import tensorflow as tf
from tensorflow import keras
import os
import warnings
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


# In[57]:


#set the best selected parameters
learning_rate = 0.01
size = 128
droprate = 0.2


# ## Model Architecture for Larger Model Training with Best Evaluated Hyperparameters

# In[81]:


# create the model architechture for the larger model with best parameters

#for larger model
input_size = 299

def final_model(input_size, learning_rate, size_inner, droprate):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    outputs = keras.layers.Dense(5)(drop)
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[83]:


# Load training data
train_ds = train_datagen.flow_from_directory(
    '/Users/sindhujaarivukkarasu/Documents/ML Zoomcamp-Capstone-2/Split_data/train',
    target_size=(299, 299),  # Adjust to match your model's input size
    batch_size=32,
    class_mode="categorical",
    shuffle=True,  # Shuffle for training
    seed=42
)

# Load validation data
val_ds = val_test_datagen.flow_from_directory(
    '/Users/sindhujaarivukkarasu/Documents/ML Zoomcamp-Capstone-2/Split_data/val',
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,  # No shuffle for validation
    seed=42
)


# ## Model Checkpointing

# In[94]:


# checkpointing to save the best model
checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v_final_{epoch:02d}_{val_accuracy:.2f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# training the larger model
print('learning_rate is ', learning_rate)
print('size is ', size)
print('droprate is ', droprate)
print()
print()

model1 = final_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

#suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*HDF5.*")

history = model1.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])


# ## Using the Model on Test Data

# In[98]:


from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create the test generator
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    '/Users/sindhujaarivukkarasu/Documents/ML Zoomcamp-Capstone-2/Split_data/test',
    target_size=(299, 299),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,  # No shuffle for validation
    seed=42
)


# In[100]:


# loading the model
model = keras.models.load_model('xception_v_final_35_0.88.keras')


# In[102]:


model.evaluate(test_ds)


# The final model shows a high accuracy of 0.92 when evaluated on the unseen test data

# In[126]:


from tensorflow.keras.models import load_model
import pickle
# Load the model
final_model = load_model('xception_v_final_35_0.88.keras')

# Save it as a pickle file
with open('grapevine_xception_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model saved as 'grapevine_xception_model.pkl'")


# In[ ]:




