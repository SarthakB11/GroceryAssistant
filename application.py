import os
import concurrent.futures
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras_cv import layers as layers_cv
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from paddleocr import PaddleOCR
import traceback
import time
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import cv2
from metaphone import doublemetaphone
from rapidfuzz import process,fuzz
import spacy
import re
from datetime import datetime
from dateutil import parser
import streamlit as st
import google.generativeai as genai

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Segment model
segment_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/full_50_2_640.pt').to(device)

# Login to Huggingface
login(token='YOUR_HUGGINGFACE_TOKEN')

instruction = (
    "You are an expert in fuzzy string matching. Given a list of product names, your task is to identify "
    "the product name that most closely matches a specified input string. Perform a strict fuzzy match, "
    "ensuring to output only one product name from the list. If no clear match is found, return 'input string'. "
    "Your output should be a single line with no extra text, explanations, or formatting."
)

# Load the pre-trained SpaCy model
nlp = spacy.load('en_core_web_sm')

# Setup Google Generative AI
genai.configure(api_key="YOUR_API_KEY")
gemma = genai.GenerativeModel('gemini-1.5-flash', system_instruction=instruction)

# Load the EfficientNetB2 model for freshness vs. rotten prediction
freshness_base_model = keras.applications.EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(260, 260, 3)
)
x = layers.Flatten()(freshness_base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)
freshness_model = keras.Model(inputs=freshness_base_model.input, outputs=predictions)

# Load the weights for the freshness vs. rotten model
freshness_model.load_weights("models/fresh_vs_rotten_v1a.weights.h5")

#Load the product vs. rotten model
model = keras.models.load_model('models/fresh_vs_rotten_product.keras')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, rec_algorithm="SVTR_LCNet", ocr_version='PP-OCRv4', use_space_char=True)

# Load the product list CSV
df = pd.read_csv("product_list.csv")

# Convert product names to lowercase and create the product list
product_list = df['Product Name'].str.lower().to_list()

# Define the word list (converted to lowercase for consistency)
word_list = ['amul malai paneer', 'borges durum wheat pasta', 'wheat pasta', 'fantastic bathroom'] + [name.lower() for name in product_list]

# Create a mapping of product names to categories
category_mapping = dict(zip(df['Product Name'].str.lower(), df['Category']))

manual_categories = {
    'amul malai paneer': 'Dairy',
    'borges durum wheat pasta': 'Pasta',
    'wheat pasta': 'Pasta',
    'fantastic bathroom': 'Home & Kitchen'
}

# Update the category_mapping with manual categories
for product, category in manual_categories.items():
    category_mapping[product] = category
# Function to get the category if a product name is found
def get_category(product_name):
    # Convert product name to lowercase
    product_name_lower = product_name.lower()
    # Check if the product name is in the mapping and return the category
    return category_mapping.get(product_name_lower, "Category not found")

# Precompute Metaphone codes for the word list
metaphone_codes = {word: doublemetaphone(word)[0] for word in word_list}

# Convert word list to set for exact matching
word_set = set(word_list)

def fuzzy_match(product_name):
    best_match, score, _ = process.extractOne(product_name, word_list, scorer=fuzz.WRatio)
    if score > 80:
        return best_match
    else:
        return "No Result found"  # Return as "No Result found"

def predict_product_name(ocr_text):
    # If the product name is more than 5 words, truncate it
    words = ocr_text.split()
    truncated_name = ' '.join(words[:5]) if len(words) > 5 else ocr_text

    # Use Generative AI to match the product name
    query = f"""
I have a list of product names: {word_list}. I want to find the product name from this list that most closely matches the following input: "{truncated_name}".

Your task is to perform a strict fuzzy match, ensuring the output is only one product name from the list. If no clear match is found, return "No Result found".

Output only the closest product name in a single line, with no extra text, explanations, or formatting. Do not include anything other than the product name itself or "No Result found" if there is no suitable match.
"""

    response = gemma.generate_content(query)
    return response.text.strip()

def calculate_overlap_percentage(xa1, xa2, ya1, ya2, xb1, xb2, yb1, yb2):
    x_overlap1 = max(xa1, xb1)
    y_overlap1 = max(ya1, yb1)
    x_overlap2 = min(xa2, xb2)
    y_overlap2 = min(ya2, yb2)
    if x_overlap1 < x_overlap2 and y_overlap1 < y_overlap2:
        overlap_area = (x_overlap2 - x_overlap1) * (y_overlap2 - y_overlap1)
    else:
        overlap_area = 0
    rect_b_area = (xb2 - xb1) * (yb2 - yb1)
    if rect_b_area == 0:
        return 0
    overlap_percentage = (overlap_area / rect_b_area) * 100
    return overlap_percentage


# def process_image(image):
    # Other existing code remains unchanged
    # Step 1: Read the image file
    image_path = 'imag1.jpg'
    image2 = cv2.imread(image_path)
    print("hi")
    # Run object detection
    results = segment_model(image2)
    predictions = results.pred[0]
    coordinates = []

    # Extract bounding box coordinates
    for *box, conf, cls in predictions:
        x1, y1, x2, y2 = map(int, box)
        coordinates.append((x1, x2, y1, y2))

    correct = []
    # Filter overlapping boxes
    i=0
    while i < len(coordinates):
        j=0
        while j < len(coordinates):
            if i != j:
                percentage = calculate_overlap_percentage(*coordinates[j], *coordinates[i])
                if percentage > 80:
                    coordinates.pop(i)
                    i-=1
                    break
            j+=1
        i+=1
    product_image_paths = []
    grocery_image_paths = []

    # Dictionary to hold products and expiry dates with counts
    products_dict = {}

    # Initialize list to hold JSON data
    products_json_list = []
    grocery_json_list = []

    # Save cropped images and predict freshness
    for i, (x1, x2, y1, y2) in enumerate(coordinates):
        cropped_image = image2[y1:y2, x1:x2]
        img_path = f'image_{i}.jpg'
        cv2.imwrite(img_path, cropped_image)
        print("hii")
        # Load and preprocess the cropped image for freshness prediction
        img = keras.utils.load_img(img_path, target_size=(260, 260))
        img_array = keras.utils.img_to_array(img)
        img_array = np.array([img_array])

        preds = model.predict(img_array)
        first_determination = np.argmax(preds)
        print(f"Predictions for image {i}: {preds}, Determination: {first_determination}")

        if first_determination == 0:
            product_image_paths.append(img_path)
        else:
            grocery_image_paths.append(img_path)
    start_time = time.time()
    # Perform OCR and process products
    for path in product_image_paths:
        text = ""
        result = ocr.ocr(path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    text += line[1][0] + " "
                print(text)

                # Predict the product name
                predicted_product_name = predict_product_name(text.strip().lower())

                # Perform fuzzy matching
                fuzzy_product_name = fuzzy_match(text.strip().lower())

                # Initialize the product_name variable
                product_name = ""

                # Check if the predicted product name is "No Result found"
                if predicted_product_name == "No Result found":
                    product_name = fuzzy_product_name  # Use the fuzzy matched product name directly
                else:
                    # Calculate the fuzzy match score
                    score = fuzz.ratio(text.strip().lower(), predicted_product_name)
                    print(score)
                    # Decide which product name to use based on the score
                    if score >= 50:
                        product_name = predicted_product_name  # Keep the predicted product name
                    else:
                        product_name = fuzzy_product_name  # Use the fuzzy matched product name if score is below 80

                # Output the final product name and score for debugging
                print(f"Predicted Product Name: {predicted_product_name}")
                print(f"Fuzzy Matched Product Name: {fuzzy_product_name} with score: {score if predicted_product_name != 'No Result found' else 'N/A'}")
                print(f"Final Product Name: {product_name}")

        # Extract dates using regex and SpaCy
        dates = []
        regex_patterns = [
            r'(?:exp(?:iry)?[: ]?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  
            r'(?:due[: ]?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',          
            r'(?:before\s?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',         
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',                     
            r'\b(\d{4}[.-]\d{1,2}[.-]\d{1,2})\b',                       
            r'\b(\d{1,2}[.-]\d{1,2}[.-]\d{4})\b',                       
            r'\b(\w{3} \d{1,2}, \d{4})\b',                              
            r'\b(\d{1,2} \w{3,9} \d{4})\b'                               
        ]
        
        for pattern in regex_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)

        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                dates.append(ent.text)

        unique_dates = list(set(dates))
        print(f"{'date'}:{unique_dates}")

        max_date = None
        if unique_dates:
            date_objects = []
            for date_str in unique_dates:
                try:
                    normalized_date = parser.parse(date_str)
                    date_objects.append(normalized_date)
                except ValueError:
                    print(f"Could not parse date: {date_str}")

            if date_objects:
                max_date = max(date_objects)


        if product_name == "No Result found":
            continue

        # Ensure the product is in products_dict
        if product_name in products_dict:
            products_dict[product_name]['count'] += 1
            if max_date:
                existing_max_date = parser.parse(products_dict[product_name]['expiry_date'])
                if max_date > existing_max_date:
                    products_dict[product_name]['expiry_date'] = max_date.strftime('%d-%m-%Y')
        else:  # If not in the dictionary
            found_same_expiry = False
            if max_date:
                for existing_product, info in products_dict.items():
                    if info['expiry_date'] == max_date.strftime('%d-%m-%Y'):
                        products_dict[existing_product]['count'] += 1
                        found_same_expiry = True
                        break

            # If no product with the same expiry date was found, add the new product
            if not found_same_expiry:
                products_dict[product_name] = {
                    'count': 1,
                    'expiry_date': max_date.strftime('%d-%m-%Y') if max_date else "No valid date found"
                }

        # Append product info to JSON list after processing all products
        # This should happen outside the loop that processes products
        if product_name in products_dict:
            category = get_category(product_name)  # Get the category using the function
            products_json_list.append({
                "product_name": product_name,
                "product_category": category,
                "count": products_dict[product_name]['count'],
                "expiry_date": products_dict[product_name]['expiry_date'],
                "image": path  # Include image path in JSON
            })

    for product,info in products_dict.items():
            print(f"Product: {product}, Count: {info['count']}, Expiry Date: {info['expiry_date']}")
            

    # Process grocery images for freshness
    for path in grocery_image_paths:
        img = keras.utils.load_img(path, target_size=(260, 260))
        img_arr = keras.utils.img_to_array(img)
        img_arr = np.array([img_arr])

        preds = model.predict(img_arr)
        freshness_index = preds[0][1] * 100
        print(f"Freshness Index : {freshness_index}")

        # Append grocery freshness info to JSON list
        grocery_json_list.append({
            "freshness_index": freshness_index,
            "image": path  # Include image path in JSON
        })
    end_time = time.time()
    difference=end_time-start_time
    # Convert lists to JSON
    products_json = json.dumps(products_json_list, indent=4)
    grocery_json = json.dumps(grocery_json_list, indent=4)

    # Save JSON files (optional)
    with open('products.json', 'w') as f:
        f.write(products_json)

    with open('grocery_freshness.json', 'w') as f:
        f.write(grocery_json)

    # Display the results
    print("Product JSON:", products_json)
    print("Grocery Freshness JSON:", grocery_json)
    # Update session state with processed data
    st.session_state.products_json_list = products_json_list
    st.session_state.grocery_json_list = grocery_json_list
    st.session_state.time_difference=difference
def process_image(image):

    # Other existing code remains unchanged
    # Step 1: Read the image file
    image_path = 'test.jpg'
    image2 = cv2.imread(image_path)
    print("hi")
    # Run object detection
    results = segment_model(image2)
    predictions = results.pred[0]
    coordinates = []

    # Extract bounding box coordinates
    for *box, conf, cls in predictions:
        x1, y1, x2, y2 = map(int, box)
        coordinates.append((x1, x2, y1, y2))

    correct = []
    # Filter overlapping boxes
    i = 0
    while i < len(coordinates):
        j = 0
        while j < len(coordinates):
            if i != j:
                percentage = calculate_overlap_percentage(*coordinates[j], *coordinates[i])
                if percentage > 80:
                    coordinates.pop(i)
                    i -= 1
                    break
            j += 1
        i += 1
    product_image_paths = []
    grocery_image_paths = []

    # Dictionary to hold products and expiry dates with counts
    products_dict = {}

    # Initialize list to hold JSON data
    products_json_list = []
    grocery_json_list = []

    # Save cropped images and predict freshness
    for i, (x1, x2, y1, y2) in enumerate(coordinates):
        cropped_image = image2[y1:y2, x1:x2]
        img_path = f'image_{i}.jpg'
        cv2.imwrite(img_path, cropped_image)
        print("hii")
        # Load and preprocess the cropped image for freshness prediction
        img = keras.utils.load_img(img_path, target_size=(260, 260))
        img_array = keras.utils.img_to_array(img)
        img_array = np.array([img_array])

        preds = model.predict(img_array)
        first_determination = np.argmax(preds)
        print(f"Predictions for image {i}: {preds}, Determination: {first_determination}")

        if first_determination == 0:
            product_image_paths.append(img_path)
        else:
            grocery_image_paths.append(img_path)

    start_time = time.time()

    # Helper function to process OCR and prediction for each image path
    def process_ocr_and_predict(path):
        text = ""
        result = ocr.ocr(path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    text += line[1][0] + " "

        # Predict the product name
        predicted_product_name = predict_product_name(text.strip().lower())

        # Perform fuzzy matching
        fuzzy_product_name = fuzzy_match(text.strip().lower())

        # Initialize the product_name variable
        product_name = ""

        # Check if the predicted product name is "No Result found"
        if predicted_product_name == "No Result found":
            product_name = fuzzy_product_name  # Use the fuzzy matched product name directly
        else:
            # Calculate the fuzzy match score
            score = fuzz.ratio(text.strip().lower(), predicted_product_name)
            # Decide which product name to use based on the score
            if score >= 50:
                product_name = predicted_product_name  # Keep the predicted product name
            else:
                product_name = fuzzy_product_name  # Use the fuzzy matched product name if score is below 80

        return product_name, text

    # Perform OCR and process products using concurrent futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(process_ocr_and_predict, path): path for path in product_image_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                product_name, text = future.result()
            except Exception as exc:
                print(f'{path} generated an exception: {exc}')
                continue

            # Extract dates using regex and SpaCy
            dates = []
            regex_patterns = [
                r'(?:exp(?:iry)?[: ]?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  
                r'(?:due[: ]?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',          
                r'(?:before\s?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',         
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',                     
                r'\b(\d{4}[.-]\d{1,2}[.-]\d{1,2})\b',                       
                r'\b(\d{1,2}[.-]\d{1,2}[.-]\d{4})\b',                       
                r'\b(\w{3} \d{1,2}, \d{4})\b',                              
                r'\b(\d{1,2} \w{3,9} \d{4})\b'                               
            ]

            for pattern in regex_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates.extend(matches)

            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'DATE':
                    dates.append(ent.text)

            unique_dates = list(set(dates))
            print(f"{'date'}:{unique_dates}")

            max_date = None
            if unique_dates:
                date_objects = []
                for date_str in unique_dates:
                    try:
                        normalized_date = parser.parse(date_str)
                        date_objects.append(normalized_date)
                    except ValueError:
                        print(f"Could not parse date: {date_str}")

                if date_objects:
                    max_date = max(date_objects)

            if product_name == "No Result found":
                continue

            # Ensure the product is in products_dict
            if product_name in products_dict:
                products_dict[product_name]['count'] += 1
                if max_date:
                    existing_max_date = parser.parse(products_dict[product_name]['expiry_date'])
                    if max_date > existing_max_date:
                        products_dict[product_name]['expiry_date'] = max_date.strftime('%d-%m-%Y')
            else:  # If not in the dictionary
                found_same_expiry = False
                if max_date:
                    for existing_product, info in products_dict.items():
                        if info['expiry_date'] == max_date.strftime('%d-%m-%Y'):
                            products_dict[existing_product]['count'] += 1
                            found_same_expiry = True
                            break

                # If no product with the same expiry date was found, add the new product
                if not found_same_expiry:
                    products_dict[product_name] = {
                        'count': 1,
                        'expiry_date': max_date.strftime('%d-%m-%Y') if max_date else "No valid date found"
                    }

            # Append product info to JSON list after processing all products
            category = get_category(product_name)  # Get the category using the function
            products_json_list.append({
                "product_name": product_name,
                "product_category": category,
                "count": products_dict[product_name]['count'],
                "expiry_date": products_dict[product_name]['expiry_date'],
                "image": path  # Include image path in JSON
            })

    for product, info in products_dict.items():
        print(f"Product: {product}, Count: {info['count']}, Expiry Date: {info['expiry_date']}")

    # Process grocery images for freshness
    for path in grocery_image_paths:
        img = keras.utils.load_img(path, target_size=(260, 260))
        img_arr = keras.utils.img_to_array(img)
        img_arr = np.array([img_arr])

        preds = model.predict(img_arr)
        freshness_index = preds[0][1] * 100
        print(f"Freshness Index : {freshness_index}")

        # Append grocery freshness info to JSON list
        grocery_json_list.append({
            "freshness_index": freshness_index,
            "image": path  # Include image path in JSON
        })
    end_time = time.time()
    difference = end_time - start_time

    # Convert lists to JSON
    products_json = json.dumps(products_json_list, indent=4)
    grocery_json = json.dumps(grocery_json_list, indent=4)

    # Save JSON files (optional)
    with open('products.json', 'w') as f:
        f.write(products_json)

    with open('grocery_freshness.json', 'w') as f:
        f.write(grocery_json)

    # Display the results
    print("Product JSON:", products_json)
    print("Grocery Freshness JSON:", grocery_json)
    # Update session state with processed data
    st.session_state.products_json_list = products_json_list
    st.session_state.grocery_json_list = grocery_json_list
    st.session_state.time_difference = difference
# def process_image(image):
    # Other existing code remains unchanged
    # Step 1: Read the image file
    image_path = 'imag1.jpg'
    image2 = cv2.imread(image_path)
    print("hi")
    # Run object detection
    results = segment_model(image2)
    predictions = results.pred[0]
    coordinates = []

    # Extract bounding box coordinates
    for *box, conf, cls in predictions:
        x1, y1, x2, y2 = map(int, box)
        coordinates.append((x1, x2, y1, y2))

    correct = []
    # Filter overlapping boxes
    i = 0
    while i < len(coordinates):
        j = 0
        while j < len(coordinates):
            if i != j:
                percentage = calculate_overlap_percentage(*coordinates[j], *coordinates[i])
                if percentage > 80:
                    coordinates.pop(i)
                    i -= 1
                    break
            j += 1
        i += 1
    product_image_paths = []
    grocery_image_paths = []

    # Dictionary to hold products and expiry dates with counts
    products_dict = {}

    # Initialize list to hold JSON data
    products_json_list = []
    grocery_json_list = []

    # Save cropped images and predict freshness
    for i, (x1, x2, y1, y2) in enumerate(coordinates):
        cropped_image = image2[y1:y2, x1:x2]
        img_path = f'image_{i}.jpg'
        cv2.imwrite(img_path, cropped_image)
        print("hii")
        # Load and preprocess the cropped image for freshness prediction
        img = keras.utils.load_img(img_path, target_size=(260, 260))
        img_array = keras.utils.img_to_array(img)
        img_array = np.array([img_array])

        preds = model.predict(img_array)
        first_determination = np.argmax(preds)
        print(f"Predictions for image {i}: {preds}, Determination: {first_determination}")

        if first_determination == 0:
            product_image_paths.append(img_path)
        else:
            grocery_image_paths.append(img_path)

    start_time = time.time()

    # Helper function to process OCR and prediction for each image path
    def process_ocr_and_predict(path):
        text = ""
        result = ocr.ocr(path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    text += line[1][0] + " "

        # Predict the product name
        predicted_product_name = predict_product_name(text.strip().lower())

        # Perform fuzzy matching
        fuzzy_product_name = fuzzy_match(text.strip().lower())

        # Initialize the product_name variable
        product_name = ""

        # Check if the predicted product name is "No Result found"
        if predicted_product_name == "No Result found":
            product_name = fuzzy_product_name  # Use the fuzzy matched product name directly
        else:
            # Calculate the fuzzy match score
            score = fuzz.ratio(text.strip().lower(), predicted_product_name)
            # Decide which product name to use based on the score
            if score >= 50:
                product_name = predicted_product_name  # Keep the predicted product name
            else:
                product_name = fuzzy_product_name  # Use the fuzzy matched product name if score is below 80

        return product_name, text

    # Perform OCR and process products using concurrent futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(process_ocr_and_predict, path): path for path in product_image_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                product_name, text = future.result()
            except Exception as exc:
                print(f'{path} generated an exception: {exc}')
                continue

            # Extract dates using regex and SpaCy
            dates = []
            regex_patterns = [
                r'(?:exp(?:iry)?[: ]?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  
                r'(?:due[: ]?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',          
                r'(?:before\s?)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',         
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',                     
                r'\b(\d{4}[.-]\d{1,2}[.-]\d{1,2})\b',                       
                r'\b(\d{1,2}[.-]\d{1,2}[.-]\d{4})\b',                       
                r'\b(\w{3} \d{1,2}, \d{4})\b',                              
                r'\b(\d{1,2} \w{3,9} \d{4})\b'                               
            ]

            for pattern in regex_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates.extend(matches)

            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'DATE':
                    dates.append(ent.text)

            unique_dates = list(set(dates))
            print(f"{'date'}:{unique_dates}")

            max_date = None
            if unique_dates:
                date_objects = []
                for date_str in unique_dates:
                    try:
                        normalized_date = parser.parse(date_str)
                        date_objects.append(normalized_date)
                    except ValueError:
                        print(f"Could not parse date: {date_str}")

                if date_objects:
                    max_date = max(date_objects)

            if product_name == "No Result found":
                continue

            # Ensure the product is in products_dict
            if product_name in products_dict:
                products_dict[product_name]['count'] += 1
                if max_date:
                    existing_max_date = parser.parse(products_dict[product_name]['expiry_date'])
                    if max_date > existing_max_date:
                        products_dict[product_name]['expiry_date'] = max_date.strftime('%d-%m-%Y')
            else:  # If not in the dictionary
                # Initialize the product in the dictionary
                products_dict[product_name] = {
                    'count': 1,
                    'expiry_date': max_date.strftime('%d-%m-%Y') if max_date else "No valid date found"
                }

            # Append product info to JSON list after processing all products
            category = get_category(product_name)  # Get the category using the function
            products_json_list.append({
                "product_name": product_name,
                "product_category": category,
                "count": products_dict[product_name]['count'],
                "expiry_date": products_dict[product_name]['expiry_date'],
                "image": path  # Include image path in JSON
            })

    for product, info in products_dict.items():
        print(f"Product: {product}, Count: {info['count']}, Expiry Date: {info['expiry_date']}")

    # Process grocery images for freshness
    for path in grocery_image_paths:
        img = keras.utils.load_img(path, target_size=(260, 260))
        img_arr = keras.utils.img_to_array(img)
        img_arr = np.array([img_arr])

        preds = model.predict(img_arr)
        freshness_index = preds[0][1] * 100
        print(f"Freshness Index : {freshness_index}")

        # Append grocery freshness info to JSON list
        grocery_json_list.append({
            "freshness_index": freshness_index,
            "image": path  # Include image path in JSON
        })
    end_time = time.time()
    difference = end_time - start_time

    # Convert lists to JSON
    products_json = json.dumps(products_json_list, indent=4)
    grocery_json = json.dumps(grocery_json_list, indent=4)

    # Save JSON files (optional)
    with open('products.json', 'w') as f:
        f.write(products_json)

    with open('grocery_freshness.json', 'w') as f:
        f.write(grocery_json)

    # Display the results
    print("Product JSON:", products_json)
    print("Grocery Freshness JSON:", grocery_json)
    # Update session state with processed data
    st.session_state.products_json_list = products_json_list
    st.session_state.grocery_json_list = grocery_json_list
    st.session_state.time_difference = difference

def main():
    st.set_page_config(layout="wide")

    # Initialize session state variables
    if 'timer_running' not in st.session_state:
        st.session_state.timer_running = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = 0.0
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'products_json_list' not in st.session_state:
        st.session_state.products_json_list = []  # To store product details
    if 'grocery_json_list' not in st.session_state:
        st.session_state.grocery_json_list = []  # To store grocery freshness details
    if 'time_difference' not in st.session_state:
        st.session_state.time_difference=0.0
    # Apply dark theme and custom styles
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .output-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
            background-color: #2E2E2E;
            margin-bottom: 20px;
        }
        .timer {
            font-size: 48px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            padding: 20px;
            background-color: #2E2E2E;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .product-container {
            display: flex;
            margin-bottom: 20px;
        }
        .product-image {
            width: 30%;
            padding: 10px;
        }
        .product-details {
            width: 70%;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Product Processing App")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Camera Feed")
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv2_img is not None:
                st.image(cv2_img, channels="BGR")
                
                if st.button("Process Image"):
                    st.session_state.timer_running = True
                    st.session_state.start_time = time.time()
                    st.session_state.processing_complete = False
                    process_image(cv2_img)  # Call the image processing function

    with col2:
        st.subheader("Processing Timer")
        timer_placeholder = st.empty()

        # Display and update timer
        if st.session_state.timer_running:
            while time.time() - st.session_state.start_time < 10:  # Simulate 10 seconds of processing
                elapsed_time = time.time() - st.session_state.start_time
                timer_placeholder.markdown(f'<div class="timer">{elapsed_time:.2f} s</div>', unsafe_allow_html=True)
                time.sleep(0.1)  # Update every 0.1 seconds
            
            st.session_state.timer_running = False
            st.session_state.processing_complete = True

        if st.session_state.processing_complete:
            final_time = st.session_state.time_difference
            timer_placeholder.markdown(f'<div class="timer">{final_time:.2f} s</div>', unsafe_allow_html=True)
            
            st.subheader("Product Details")
            
            # Create a dictionary to consolidate products with the highest count
            consolidated_products = {}

            # Consolidate products based on product_name and keep the one with the highest count
            for product in st.session_state.products_json_list:
                product_name = product['product_name']
                if product_name in consolidated_products:
                    # If the product already exists, update to the one with the higher count
                    if product['count'] > consolidated_products[product_name]['count']:
                        consolidated_products[product_name] = product
                else:
                    # If it does not exist, add it to the dictionary
                    consolidated_products[product_name] = product

            # Now, display the consolidated products
            for product in consolidated_products.values():
                # Open the image file
                img = Image.open(product['image'])  # Load the image from the path
                
                # Create two columns: one for the image and one for the product details
                col1, col2 = st.columns([1, 2])  # Adjust the ratios as needed

                with col1:
                    # Display the image using Streamlit's st.image
                    st.image(img, caption=product['product_name'], use_column_width=True)  # Display the image

                with col2:
                    # Display product details
                    st.markdown(
                        f"""
                        <div class="product-details">
                            <h2>{product['product_name']}</h2>
                            <h3>{product['product_category']}</h3>
                            <p><strong>Count:</strong> {product['count']}</p>
                            <p><strong>Expiry Date:</strong> {product['expiry_date']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.subheader("Grocery Freshness Details")

            for grocery in st.session_state.grocery_json_list:
                # Open the image file
                img = Image.open(product['image'])  # Load the image from the path
                
                # Create two columns: one for the image and one for the product details
                col1, col2 = st.columns([1, 2])  # Adjust the ratios as needed

                with col1:
                    # Display the image using Streamlit's st.image
                    st.image(img, caption=product['product_name'], use_column_width=True)  # Display the image

                with col2:
                    # Display product details
                    st.markdown(
                        f"""
                            <div class="product-details">
                                <p><strong>Freshness Index:</strong> {grocery['freshness_index']:.2f}%</p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
if __name__ == "__main__":
    main()
