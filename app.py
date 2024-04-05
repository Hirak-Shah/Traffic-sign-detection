import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the model
model = load_model("my_model.h5")

# Dictionary to label all traffic signs classes
classes = { 
    1:'Speed limit (20km/h)',
    2:'Speed limit (30km/h)',      
    3:'Speed limit (50km/h)',       
    4:'Speed limit (60km/h)',      
    5:'Speed limit (70km/h)',    
    6:'Speed limit (80km/h)',      
    7:'End of speed limit (80km/h)',     
    8:'Speed limit (100km/h)',    
    9:'Speed limit (120km/h)',     
    10:'No passing',   
    11:'No passing veh over 3.5 tons',     
    12:'Right-of-way at intersection',     
    13:'Priority road',    
    14:'Yield',     
    15:'Stop',       
    16:'No vehicles',       
    17:'Veh > 3.5 tons prohibited',       
    18:'No entry',       
    19:'General caution',     
    20:'Dangerous curve left',      
    21:'Dangerous curve right',   
    22:'Double curve',      
    23:'Bumpy road',     
    24:'Slippery road',       
    25:'Road narrows on the right',  
    26:'Road work',    
    27:'Traffic signals',      
    28:'Pedestrians',     
    29:'Children crossing',     
    30:'Bicycles crossing',       
    31:'Beware of ice/snow',
    32:'Wild animals crossing',      
    33:'End speed + passing limits',      
    34:'Turn right ahead',     
    35:'Turn left ahead',       
    36:'Ahead only',      
    37:'Go straight or right',      
    38:'Go straight or left',      
    39:'Keep right',     
    40:'Keep left',      
    41:'Roundabout mandatory',     
    42:'End of no passing',      
    43:'End no passing veh > 3.5 tons' 
}

# Streamlit app script
st.title('Traffic Sign Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert image to RGB mode if it has four channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize image
    image = np.array(image.resize((30,30)))
    
    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)

    # Make prediction
    pred = model.predict(image)
    sign = classes[np.argmax(pred) + 1]

    st.write(f"Prediction: {sign}")
