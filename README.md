# NutriScan: A Receipt-Based Dietary Insight Tool Using Computer Vision

A comprehensive Streamlit-based application that scans grocery receipts using Computer Vision techniques to extract items and provide nutritional analysis.

## Features

- **Receipt Image Processing**: Upload or capture receipt images via webcam
- **Multiple CV Techniques**:
  - Image Preprocessing (grayscale conversion, adaptive thresholding, morphological operations)
  - Edge Detection & Contour Detection (Canny edge detection, perspective transform)
  - OCR Text Extraction (Tesseract-based text recognition)
- **Nutritional Analysis**: Automatic lookup of extracted items against nutrition database
- **Visual Results**: Annotated receipt images with bounding boxes highlighting recognized items
- **Summary Insights**: Total calories and macronutrient breakdown

## Installation

1. Clone the repository and navigate to the project directory
2. Install Python 3.9+
3. Install Tesseract OCR:
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Project Structure

```
nutriscan/
├── app.py                        # Main Streamlit app
├── cv_pipeline/
│   ├── __init__.py
│   ├── preprocessor.py           # Grayscale, thresholding, morphological ops
│   ├── edge_contour.py           # Canny edge detection, contour finding, perspective warp
│   └── ocr_extractor.py          # Tesseract OCR wrapper
├── parser/
│   ├── __init__.py
│   └── receipt_parser.py         # Regex logic to extract items & prices
├── nutrition/
│   ├── __init__.py
│   ├── nutrition_lookup.py       # Keyword-to-nutrient matching
│   └── nutrition_data.csv        # Common grocery items with nutritional data
├── utils/
│   ├── __init__.py
│   ├── image_utils.py            # Image conversion helpers
│   └── annotation.py             # Bounding box annotation
├── tests/
│   ├── test_preprocessor.py
│   ├── test_parser.py
│   └── test_nutrition.py
└── requirements.txt
```

## Key CV Techniques

### 1. Image Preprocessing
- Converts receipt images to grayscale
- Applies adaptive thresholding to handle uneven lighting
- Uses morphological operations (dilation/erosion) to clean noise

### 2. Edge & Contour Detection
- Canny edge detection to find receipt borders
- Contour detection and perspective transform for deskewing
- Improves OCR accuracy on skewed images

### 3. OCR Text Extraction
- Tesseract-based optical character recognition
- Extracts food item names and prices
- Regex parsing to isolate structured data

## Limitations

- OCR accuracy depends on receipt image quality
- Nutrition data is approximate (dictionary-based, not a real database)
- Best results with clear thermal receipts; handwritten receipts not supported
- Limited to ~50 common grocery items in the initial database

## Future Enhancements

- Integration with real nutrition APIs (USDA FoodData Central)
- Multi-language OCR support
- Barcode scanning integration
- Calorie tracking and dietary goal management
- Receipt history and trends analysis
