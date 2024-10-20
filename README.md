# GroceryAssistant-Flipkart-Grid-6.0

## Overview

This application is an advanced product recognition and freshness assessment system designed for inventory management and quality control in retail environments. It utilizes computer vision, optical character recognition (OCR), and machine learning techniques to process images of products and produce detailed information about them.

## Key Features

1. **Product Recognition**: Identifies products in images using object detection and OCR.
2. **Expiry Date Extraction**: Extracts and normalizes expiry dates from product labels.
3. **Freshness Assessment**: Evaluates the freshness of grocery items.
4. **Category Classification**: Automatically categorizes products based on a predefined list.
5. **Real-time Processing**: Includes a timer to track processing duration.
6. **User-friendly Interface**: Built with Streamlit for an interactive web application experience.

## Technical Approach

### 1. Image Segmentation and Object Detection

- Utilizes a custom YOLOv5 model (`full_50_2_640.pt`) for object detection.
- Implements an algorithm to filter out overlapping bounding boxes.

### 2. Product Classification

- Uses a pre-trained EfficientNetB2 model to classify images as products or groceries.

### 3. Optical Character Recognition (OCR)

- Employs PaddleOCR for text extraction from product images.

### 4. Product Name Prediction

- Implements a fuzzy string matching algorithm using RapidFuzz.
- Utilizes a generative AI model (Gemini 1.5 Flash) for improved product name prediction.

### 5. Date Extraction

- Uses regex patterns and SpaCy's named entity recognition to extract and normalize dates.

### 6. Freshness Assessment

- Applies a separate model to assess the freshness of grocery items.

### 7. Concurrent Processing

- Utilizes `concurrent.futures` for parallel processing of multiple images.

### 8. User Interface

- Built with Streamlit, featuring:
  - Live camera feed for image capture
  - Real-time processing timer
  - Detailed display of processed results

## Dependencies

- TensorFlow and Keras
- OpenCV
- PaddleOCR
- SpaCy
- RapidFuzz
- Streamlit
- Google GenerativeAI
- PyTorch
- Transformers (Hugging Face)

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download necessary model weights and place them in the `models/` directory
4. Set up environment variables for API keys (Hugging Face, Google GenerativeAI)

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

Navigate to the provided local URL in your web browser to access the application.

## Future Improvements

1. Implement more robust error handling and logging
2. Optimize processing speed for larger batches of images
3. Expand the product database and improve category classification
4. Enhance the UI for better visualization of results
5. Implement user authentication and data persistence

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
