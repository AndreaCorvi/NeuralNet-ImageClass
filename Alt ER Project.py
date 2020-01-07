#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Libraries Needed
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
import cv2
from PIL import Image
from matplotlib import patches as patches
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import keras as keras
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.applications.densenet import DenseNet121
from keras.utils import Sequence
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.models import Sequential, Model
import os


# In[3]:


data_path = "C:/Users/corvi/Desktop/ER Project/understanding_cloud_organization"
train_csv_path = "C:/Users/corvi/Desktop/ER Project/understanding_cloud_organization/train.csv"
train_image_path = "C:/Users/corvi/Desktop/ER Project/understanding_cloud_organization/train_images"


# In[4]:


train_df = pd.read_csv(r"C:\Users\corvi\Desktop\ER Project\understanding_cloud_organization\train.csv") #csv loading
print(train_df.head(2)) #looking at first two lines. Inside label we find both the reference to the image + 
                        #the actual label that's going to be our target


# In[4]:


train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1]) #Have to separate Image_Label into the id of the
train_df['Image'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0]) #image and the target variable


# In[5]:


print(train_df.isnull().sum(axis = 0)) #Check Nas, regarding 22184 images, 10348 are missing encoded pixels
print(f'There are {train_df.shape[0]} records in train.csv') 


# In[8]:


plt.figure( figsize=(50,40) )
train_df.loc[train_df['EncodedPixels'].isnull(), 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts().plot(kind="bar", color="#033362")
print("Value Counts without NAs")


# In[7]:


train_df["Image_Label"].apply(lambda x: x.split('_')[1]).value_counts().plot(kind="bar", color="orange")
print("Value Counts with Na")


# Without the observations missing the encoded pixels, the dataset seems more imbalanced.

# In[8]:


train_df['EncodedPixels'] = train_df['EncodedPixels'].fillna(0) # substitute Nas with 0s and then I remove all the rows with 0s
train_df = train_df[train_df.EncodedPixels != 0]


# In[9]:


train_df["Image_Label"].apply(lambda x: x.split('_')[0]).value_counts().value_counts().plot(kind="bar", color= "orange")
print("Number of clouds per image")


# Now we need to understand better about encoded pixel and our input

# In[10]:


for col in train_df.columns: 
    print(col) 


# In[11]:


train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)


# In[12]:


def rle_to_mask(rle_string, height, width):
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img


# In[13]:


img = cv2.imread(os.path.join(train_image_path, train_df['Image'][0]))


# In[14]:


mask_decoded = rle_to_mask(train_df['Label_EncodedPixels'][0][1], img.shape[0], img.shape[1])
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
ax[0].imshow(img)
ax[1].imshow(mask_decoded)


# So basically we got a set of satellite images and a set of corresponding masks, roughly outlining the area of the image with some kind of pattern.

# In[15]:


def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def plot_cloud(img_path, img_id, label_mask):
    img = cv2.imread(os.path.join(img_path, img_id))
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    ax[0].imshow(img)
    ax[1].imshow(img)
    cmaps = {'Fish': 'Blues', 'Flower': 'Reds', 'Gravel': 'Greys', 'Sugar':'Purples'}
    colors = {'Fish': 'Blue', 'Flower': 'Red', 'Gravel': 'Gray', 'Sugar':'Purple'}
    for label, mask in label_mask:
        mask_decoded = rle_to_mask(mask, img.shape[0], img.shape[1])
        if mask != -1:
            rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
            bbox = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor=colors[label],facecolor='none')
            ax[0].add_patch(bbox)
            ax[0].text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))
            ax[1].imshow(mask_decoded, alpha=0.3, cmap=cmaps[label])
            ax[0].text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))
            
grouped_EncodedPixels = train_df.groupby('Image')['Label_EncodedPixels'].apply(list)


# In[16]:


for image_id, label_mask in grouped_EncodedPixels.sample(5).iteritems():
    plot_cloud(train_image_path, image_id, label_mask)


# In[17]:


corr_df = pd.get_dummies(train_df, columns = ['Label'])
# fill null values with '-1'
corr_df = corr_df.fillna('-1')

# define a helper function to fill dummy columns
def get_dummy_value(row, cloud_type):
    ''' Get value for dummy column '''
    if cloud_type == 'fish':
        return row['Label_Fish'] * (row['EncodedPixels'] != '-1')
    if cloud_type == 'flower':
        return row['Label_Flower'] * (row['EncodedPixels'] != '-1')
    if cloud_type == 'gravel':
        return row['Label_Gravel'] * (row['EncodedPixels'] != '-1')
    if cloud_type == 'sugar':
        return row['Label_Sugar'] * (row['EncodedPixels'] != '-1')
    
# fill dummy columns
corr_df['Label_Fish'] = corr_df.apply(lambda row: get_dummy_value(row, 'fish'), axis=1)
corr_df['Label_Flower'] = corr_df.apply(lambda row: get_dummy_value(row, 'flower'), axis=1)
corr_df['Label_Gravel'] = corr_df.apply(lambda row: get_dummy_value(row, 'gravel'), axis=1)
corr_df['Label_Sugar'] = corr_df.apply(lambda row: get_dummy_value(row, 'sugar'), axis=1)

# check the result
corr_df.head()


# In[18]:


corr_df = corr_df.groupby('Image')['Label_Fish', 'Label_Flower', 'Label_Gravel', 'Label_Sugar'].max()
corr_df.head()


# In[19]:


corrs = np.corrcoef(corr_df.values.T)
sns.set(font_scale=1)
sns.set(rc={'figure.figsize':(7,7)})
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = ['Fish', 'Flower', 'Gravel', 'Sugar'], 
               xticklabels = ['Fish', 'Flower', 'Gravel', 'Sugar']).set_title('Cloud type correlation heatmap')

fig = hm.get_figure()


# There is not a strong correlation between the types of clouds, correlation coefficients are close to zero.

# In[20]:


def get_mask_cloud(img_path, img_id, label, mask):
    img = cv2.imread(os.path.join(img_path, img_id), 0)
    mask_decoded = rle_to_mask(mask, img.shape[0], img.shape[1])
    mask_decoded = (mask_decoded > 0.0).astype(int)
    img = np.multiply(img, mask_decoded)
    return img

def draw_label_only(label):
    samples_df = train_df[(train_df['EncodedPixels']!=-1) & (train_df['Label']==label)].sample(2)
    count = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    for idx, sample in samples_df.iterrows():
        img = get_mask_cloud(train_image_path, sample['Image'], sample['Label'],sample['EncodedPixels'])
        ax[count].imshow(img, cmap="gray")
        count += 1
    plt.grid(None)


# In[21]:


draw_label_only('Fish')
print("Fish Cloud")


# In[22]:


draw_label_only('Sugar')
print("Sugar Cloud")


# In[23]:


draw_label_only('Gravel')
print("Gravel Cloud")


# In[24]:


draw_label_only('Flower')
print("Flower Cloud")


# Clouds type are recognisible for the human eye. Of course it's not so easy for a computer, specifically it will be challenging in eventual "overlapping" areas.

# Now we can start building a classifier trained on the training set.

# In[25]:


train_img = img.copy()


# So "train_img" is my array containing my train images, so basically my x_train.
# Now I build the y_train and then x_test and y_test.

# In[26]:


for i in train_df["Label"]:
    train_df[i] = train_df['Label'].map(lambda x: 1 if i in x else 0)
train_df.head()


# For now we only split into train and validation set.

# In[27]:


columns = ["Fish","Flower","Sugar","Gravel"]
print(type(columns))


# In[34]:


train_generator = ImageDataGenerator(rescale=1./255.)
train_gen= train_generator.flow_from_dataframe(
dataframe=train_df[:7500],
directory=train_image_path,
x_col="Image",
y_col=columns,
batch_size=32,
width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
seed=50,
shuffle=True,
class_mode="other",
)


# In[35]:


val_generator = ImageDataGenerator(rescale=1./255.)
val_gen= val_generator.flow_from_dataframe(
dataframe=train_df[7500:],
directory=train_image_path,
x_col="Image",
y_col=columns,
batch_size=32,
width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
seed=50,
shuffle=True,
class_mode="other")


# In[30]:


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('sigmoid'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])


# In[31]:


STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size


# In[33]:


model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=3,
                    use_multiprocessing=False,
                    workers=6
                    )


# In[ ]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2
def get_model():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    y_pred = Dense(4, activation='sigmoid')(x)  
    return Model(inputs=base_model.input, outputs=y_pred)

model = get_model()


# In[ ]:


STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size


# In[ ]:


for base_layer in model.layers[:-1]:
    base_layer.trainable = True
    
model.compile(optimizer=Adam(lr=0.0005), loss='binary_crossentropy',metrics=["accuracy"])
history_2 = model.fit_generator(generator=train_gen,
                              validation_data=val_gen,
                              epochs=20,
                              workers=10,
                              verbose=1,
                              )


# In[ ]:





# In[ ]:




