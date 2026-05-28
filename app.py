"""
NutriScan: A Receipt-Based Dietary Insight Tool Using Computer Vision

Main Streamlit Application

This application combines Computer Vision techniques to:
1. Preprocess receipt images (grayscale, thresholding, morphological ops)
2. Detect and crop receipt boundaries (Canny edge detection, perspective transform)
3. Extract text using OCR (Tesseract)
4. Parse items and prices from text
5. Look up nutritional information
6. Visualize results and provide dietary insights

Author: CMSC 191 Final Project
Date: 2024
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Import our custom modules
from cv_pipeline.preprocessor import preprocess
from cv_pipeline.edge_contour import detect_and_warp, get_contour_outline
from cv_pipeline.ocr_extractor import extract_text, get_text_regions, get_ocr_metadata
from parser.receipt_parser import parse_items, clean_item_name
from nutrition.nutrition_lookup import enrich_parsed_items, get_nutrition_summary
from utils.image_utils import pil_to_cv, cv_to_pil, resize_for_display, get_image_info
from utils.annotation import annotate_receipt, create_annotation_legend


# ==================== Page Configuration ====================

st.set_page_config(
    page_title="NutriScan - Receipt Scanner",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom Styling ====================

st.markdown("""
<style>
    .title-main {
        color: #2ECC71;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #7F8C8D;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #ECF0F1;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2ECC71;
    }
    .warning-box {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #D4EDDA;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28A745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Sidebar Configuration ====================

with st.sidebar:
    st.title("📋 NutriScan Control Panel")
    
    # Description
    with st.expander("ℹ️ About NutriScan", expanded=False):
        st.markdown("""
        **NutriScan** is a Computer Vision application that:
        
        1. **Scans receipts** - Upload or capture receipt images
        2. **Processes images** - Applies 3 CV techniques:
           - Image preprocessing (grayscale, adaptive thresholding, morphological ops)
           - Edge detection & contour finding (Canny, perspective transform)
           - OCR text extraction (Tesseract)
        3. **Parses items** - Extracts food items and prices
        4. **Analyzes nutrition** - Looks up nutritional data
        5. **Provides insights** - Shows calorie and macronutrient breakdowns
        """)
    
    # Color legend
    with st.expander("🎨 Annotation Legend", expanded=False):
        legend = create_annotation_legend()
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Green boxes:**")
            st.write("Recognized food items")
        with col2:
            st.write("**Gray boxes:**")
            st.write("Unrecognized text")
    
    # How to use
    with st.expander("🔧 How to Use", expanded=False):
        st.markdown("""
        1. **Upload or capture** a receipt image
        2. **Click Process** to apply CV techniques
        3. **View CV steps** to see preprocessing, edge detection, and OCR
        4. **Check results** for extracted items and nutrition
        5. **Review summary** for total nutrition analysis
        """)
    
    # Processing options
    st.subheader("⚙️ Processing Options")
    
    show_preprocessing = st.checkbox("Show preprocessing steps", value=True)
    show_ocr_regions = st.checkbox("Show OCR text regions", value=True)
    show_annotated = st.checkbox("Show annotated receipt", value=True)
    
    # Information section
    st.divider()
    st.info("""
    **Note:** OCR accuracy depends on receipt image quality. 
    Best results with clear, thermal receipts.
    """)


# ==================== Main Application ====================

# Header
st.markdown('<p class="title-main">📸 NutriScan</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">A Receipt-Based Dietary Insight Tool Using Computer Vision</p>',
    unsafe_allow_html=True
)

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Capture",
    "🔬 CV Processing",
    "📊 Results",
    "📈 Nutrition Analysis"
])

# ==================== TAB 1: Upload & Capture ====================

with tab1:
    st.header("Upload or Capture Receipt")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a receipt image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="receipt_uploader"
        )
    
    with col2:
        st.subheader("📷 Capture with Camera")
        camera_image = st.camera_input("Take a receipt photo")
    
    # Select which input to use
    image_input = None
    image_source = None
    
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)
        image_source = "upload"
    elif camera_image is not None:
        image_input = Image.open(camera_image)
        image_source = "camera"
    
    if image_input is not None:
        # Store in session state
        st.session_state.original_image = image_input
        st.session_state.image_source = image_source
        
        # Display original image
        st.success(f"✓ Image loaded ({image_source})")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image_input, caption="Original Receipt Image", use_column_width=True)
        
        with col2:
            # Image info
            img_array = np.array(image_input)
            info = get_image_info(pil_to_cv(image_input))
            st.metric("Width", f"{info['width']}px")
            st.metric("Height", f"{info['height']}px")
            st.metric("Channels", info['channels'])
            st.metric("Size", f"{info['size_mb']:.2f}MB")
        
        # Process button
        if st.button("🚀 Process Receipt", key="process_btn", use_container_width=True):
            st.session_state.process_triggered = True


# ==================== TAB 2: CV Processing ====================

with tab2:
    st.header("Computer Vision Processing Pipeline")
    
    if 'original_image' not in st.session_state:
        st.warning("⚠️ Please upload or capture a receipt image first (Upload & Capture tab)")
    else:
        if st.session_state.get('process_triggered', False):
            with st.spinner("🔄 Processing receipt..."):
                # Convert PIL to OpenCV format
                original_cv = pil_to_cv(st.session_state.original_image)
                
                # ========== TECHNIQUE 1: Preprocessing ==========
                st.subheader("Technique 1️⃣: Image Preprocessing")
                st.markdown("""
                **What it does:** Converts image to grayscale, applies adaptive thresholding 
                to handle uneven lighting, and uses morphological operations to clean noise.
                """)
                
                preprocessed, gray, thresh = preprocess(original_cv)
                st.session_state.preprocessed_image = preprocessed
                
                if show_preprocessing:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(cv_to_pil(gray), caption="Step 1: Grayscale", use_column_width=True)
                    
                    with col2:
                        st.image(cv_to_pil(thresh), caption="Step 2: Adaptive Thresholding", use_column_width=True)
                    
                    with col3:
                        st.image(cv_to_pil(preprocessed), caption="Step 3: Morphological Cleanup", use_column_width=True)
                else:
                    st.info("✓ Preprocessing completed (visualization hidden)")
                
                st.divider()
                
                # ========== TECHNIQUE 2: Edge Detection & Contour Finding ==========
                st.subheader("Technique 2️⃣: Edge Detection & Contour Finding")
                st.markdown("""
                **What it does:** Uses Canny edge detection to find receipt borders, 
                detects contours, and applies perspective transformation for deskewing.
                """)
                
                warped, contour_image = detect_and_warp(original_cv, preprocessed)
                st.session_state.warped_image = warped
                st.session_state.contour_image = contour_image
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(cv_to_pil(contour_image), caption="Detected Contours", use_column_width=True)
                
                with col2:
                    st.image(cv_to_pil(warped), caption="Perspective Transformed", use_column_width=True)
                
                st.divider()
                
                # ========== TECHNIQUE 3: OCR Text Extraction ==========
                st.subheader("Technique 3️⃣: OCR Text Extraction")
                st.markdown("""
                **What it does:** Uses Tesseract OCR to extract text from the 
                preprocessed receipt image.
                """)
                
                with st.spinner("🔤 Extracting text with Tesseract OCR..."):
                    raw_text = extract_text(warped)
                    st.session_state.raw_ocr_text = raw_text
                    
                    # Get text regions
                    ocr_metadata = get_ocr_metadata(warped)
                    text_regions = ocr_metadata['text_regions']
                    st.session_state.text_regions = text_regions
                
                # Display OCR results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Extracted Text:**")
                    st.text_area(
                        "Raw OCR Output",
                        value=raw_text,
                        height=150,
                        disabled=True,
                        key="ocr_text_display"
                    )
                
                with col2:
                    st.metric("Text Regions Found", len(text_regions))
                    st.metric("Avg. OCR Confidence", f"{ocr_metadata['avg_confidence']:.1f}%")
                
                # Show annotated receipt with text regions
                if show_ocr_regions and len(text_regions) > 0:
                    st.write("**Detected Text Regions (with confidence):**")
                    annotated = cv_to_pil(original_cv)
                    from utils.annotation import draw_bounding_boxes
                    annotated_cv = draw_bounding_boxes(original_cv, text_regions, color=(0, 255, 0), thickness=2)
                    st.image(cv_to_pil(annotated_cv), caption="Text Regions Detected", use_column_width=True)
                
                st.success("✅ CV Pipeline Processing Complete!")


# ==================== TAB 3: Results ====================

with tab3:
    st.header("Receipt Analysis Results")
    
    if 'raw_ocr_text' not in st.session_state:
        st.warning("⚠️ Please process a receipt first (CV Processing tab)")
    else:
        # Parse items from OCR text
        with st.spinner("📝 Parsing receipt items..."):
            parsed_items = parse_items(st.session_state.raw_ocr_text)
            st.session_state.parsed_items = parsed_items
        
        # Enrich with nutrition data
        with st.spinner("🥗 Looking up nutritional data..."):
            enriched_items = enrich_parsed_items(parsed_items)
            st.session_state.enriched_items = enriched_items
        
        # Display parsed items
        if len(parsed_items) > 0:
            st.subheader("📋 Extracted Items")
            
            # Create display dataframe
            display_items = []
            for item in enriched_items:
                display_items.append({
                    'Item Name': clean_item_name(item.get('name', 'Unknown')),
                    'Price': f"${item.get('price', 0):.2f}",
                    'Calories': int(item.get('calories', 0)),
                    'Status': "✓ Found" if item.get('found') else "❌ Not in DB"
                })
            
            st.dataframe(display_items, use_container_width=True, hide_index=True)
            
            # Show recognized items for annotation
            recognized_items = {item.get('name', '') for item in enriched_items if item.get('found')}
            
            if show_annotated and recognized_items:
                st.subheader("🎨 Annotated Receipt")
                annotated_img = annotate_receipt(
                    st.session_state.original_image if hasattr(st.session_state, 'original_image') else None,
                    st.session_state.get('text_regions', []),
                    recognized_items
                )
                if annotated_img is not None:
                    st.image(annotated_img, caption="Receipt with Recognized Items Highlighted", use_column_width=True)
        
        else:
            st.warning("⚠️ No items could be parsed from the receipt.")
            st.info("Try with a clearer receipt image or ensure the receipt is straight.")


# ==================== TAB 4: Nutrition Analysis ====================

with tab4:
    st.header("📊 Nutritional Analysis & Insights")
    
    if 'enriched_items' not in st.session_state or len(st.session_state.enriched_items) == 0:
        st.warning("⚠️ Please process a receipt first (Tabs 1-3)")
    else:
        enriched_items = st.session_state.enriched_items
        
        # Calculate nutrition summary
        nutrition_summary = get_nutrition_summary(enriched_items)
        
        # Display key metrics
        st.subheader("🎯 Nutritional Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Calories",
                f"{nutrition_summary['total_calories']:.0f} kcal",
                delta=f"{nutrition_summary['items_count']} items"
            )
        
        with col2:
            st.metric(
                "Total Protein",
                f"{nutrition_summary['total_protein_g']:.1f}g",
                delta="from " + str(nutrition_summary['items_count']) + " items"
            )
        
        with col3:
            st.metric(
                "Total Carbs",
                f"{nutrition_summary['total_carbs_g']:.1f}g"
            )
        
        with col4:
            st.metric(
                "Total Fat",
                f"{nutrition_summary['total_fat_g']:.1f}g"
            )
        
        st.divider()
        
        # Macronutrient breakdown
        st.subheader("🥗 Macronutrient Breakdown")
        
        if nutrition_summary['total_calories'] > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Pie chart for macros
                macro_data = {
                    'Protein': nutrition_summary.get('protein_percentage', 0),
                    'Carbs': nutrition_summary.get('carbs_percentage', 0),
                    'Fat': nutrition_summary.get('fat_percentage', 0)
                }
                
                # Create bar chart using st.bar_chart
                macro_df = []
                for macro, percentage in macro_data.items():
                    macro_df.append({'Macronutrient': macro, 'Percentage': percentage})
                
                import pandas as pd
                df_macro = pd.DataFrame(macro_df)
                
                st.bar_chart(
                    df_macro.set_index('Macronutrient'),
                    use_container_width=True
                )
            
            with col2:
                st.info(f"""
                **Macronutrient Percentages:**
                - Protein: {nutrition_summary.get('protein_percentage', 0):.1f}%
                - Carbs: {nutrition_summary.get('carbs_percentage', 0):.1f}%
                - Fat: {nutrition_summary.get('fat_percentage', 0):.1f}%
                """)
        
        st.divider()
        
        # Items found vs not found
        st.subheader("📈 Database Match Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Items Found in DB",
                nutrition_summary['found_items'],
                delta=f"+{nutrition_summary['found_items']}"
            )
        
        with col2:
            st.metric(
                "Items Not Found",
                nutrition_summary['not_found_items']
            )
        
        with col3:
            if nutrition_summary['items_count'] > 0:
                found_percentage = (nutrition_summary['found_items'] / nutrition_summary['items_count']) * 100
                st.metric(
                    "Match Rate",
                    f"{found_percentage:.0f}%"
                )
        
        st.divider()
        
        # Detailed items table
        st.subheader("📋 Detailed Items Breakdown")
        
        detail_items = []
        for item in enriched_items:
            detail_items.append({
                'Item': clean_item_name(item.get('name', 'Unknown')),
                'Price': f"${item.get('price', 0):.2f}",
                'Calories': f"{int(item.get('calories', 0))} kcal",
                'Protein': f"{item.get('protein_g', 0):.1f}g",
                'Carbs': f"{item.get('carbs_g', 0):.1f}g",
                'Fat': f"{item.get('fat_g', 0):.1f}g",
                'Status': "✓ DB" if item.get('found') else "⚠️ Estimate"
            })
        
        st.dataframe(detail_items, use_container_width=True, hide_index=True)
        
        # Recommendations
        st.subheader("💡 Dietary Insights")
        
        if nutrition_summary['total_calories'] > 2500:
            st.warning(f"⚠️ High calorie intake: {nutrition_summary['total_calories']:.0f} kcal")
        elif nutrition_summary['total_calories'] > 2000:
            st.info(f"ℹ️ Moderate calorie intake: {nutrition_summary['total_calories']:.0f} kcal")
        else:
            st.success(f"✓ Reasonable calorie intake: {nutrition_summary['total_calories']:.0f} kcal")
        
        # Protein recommendation (general guideline: 0.8-1g per kg bodyweight, assume 70kg)
        if nutrition_summary['total_protein_g'] >= 56:
            st.success(f"✓ Good protein intake: {nutrition_summary['total_protein_g']:.1f}g")
        else:
            st.info(f"ℹ️ Protein: {nutrition_summary['total_protein_g']:.1f}g")
        
        # Export results
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Results as CSV"):
                # Create CSV data
                csv_data = "Item Name,Price,Calories,Protein(g),Carbs(g),Fat(g),Status\n"
                for item in enriched_items:
                    csv_data += f"{clean_item_name(item.get('name', ''))},${item.get('price', 0):.2f},{int(item.get('calories', 0))},{item.get('protein_g', 0):.1f},{item.get('carbs_g', 0):.1f},{item.get('fat_g', 0):.1f},{'Found' if item.get('found') else 'Not Found'}\n"
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="nutriscan_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Clear All Data"):
                st.session_state.clear()
                st.success("✓ All data cleared. Refresh to start over.")


# ==================== Footer ====================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.write("**🎓 CMSC 191: Computer Vision**")
    st.write("Final Project")

with footer_col2:
    st.write("**📚 CV Techniques Used:**")
    st.write("✓ Image Preprocessing")
    st.write("✓ Edge/Contour Detection")
    st.write("✓ OCR Text Extraction")

with footer_col3:
    st.write("**⚙️ Technology Stack:**")
    st.write("• OpenCV")
    st.write("• Tesseract")
    st.write("• Streamlit")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("NutriScan © 2024 | A Receipt-Based Dietary Insight Tool Using Computer Vision")
