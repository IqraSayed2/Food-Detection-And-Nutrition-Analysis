
import re
import cv2
import torch
import numpy as np
from pathlib import Path
from tabulate import tabulate
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class llamaOutput:
    def __init__(self ):
        
        # Use a public, open-access CausalLM model from Hugging Face
        self.base_model = 'distilgpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,

            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",  # Use Accelerate to handle devices
            trust_remote_code=True,
        )

        # Set pad_token_id if not already set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            # model = 'llama-3.2-transformers-1b-instruct-v1',
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )



    def nutrition_from_yolo_pred_class(self, yolo_clas):
        """
        Fetch nutrition info dynamically from Open Food Facts API for the detected class.
        Normalize class name and increase timeout for reliability. Retry on failure.
        """
        import time
        normalized = yolo_clas.lower()
        for suffix in ["-fresh", "-raw", "-green", "-salad", "-leaf", "-cooked", "-boiled", "-steamed", "-grilled", "-baked", "-fried"]:
            if normalized.endswith(suffix):
                normalized = normalized.replace(suffix, "")
        normalized = normalized.replace("-", " ").replace("_", " ").strip()
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={normalized}&search_simple=1&action=process&json=1&page_size=1"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                data = response.json()
                if data.get('products'):
                    product = data['products'][0]
                    nutriments = product.get('nutriments', {})
                    nutrition_lines = []
                    for key, value in nutriments.items():
                        if any(x in key for x in ['_100g', 'energy', 'fat', 'carbohydrates', 'proteins', 'fiber', 'sugars', 'salt', 'sodium']):
                            nutrition_lines.append(f"{key}: {value}")
                    if nutrition_lines:
                        return '\n'.join(nutrition_lines)
                    else:
                        return "No nutrition data found for this food."
                else:
                    return "No product found for this food class."
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                return f"Error fetching nutrition info: {str(e)}"



    def parse_nutrition_data(self, text):
        """
        Parse nutrition data from Open Food Facts API (key: value per line).
        """
        nutrition_data = []
        for line in text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                nutrition_data.append([key.strip(), value.strip()])
        if not nutrition_data:
            return "No nutrition data found."
        headers = ["Nutrient/Component", "Value"]
        table = tabulate(nutrition_data, headers=headers, tablefmt="grid")
        return table


    def put_nutrition_and_save(self, processed_img, nutrition, output_directory, image_name,class_name ):
        """
        Overlay nutrition information on an image as a smaller, semi-transparent table and save it.

        Args:
            processed_img (numpy.ndarray): The image to annotate.
            nutrition (str): Nutrition information as a multi-line string.
            output_directory (str): Path to save the processed image.
            image_name (str): Name of the image file.
            font: OpenCV font type.
            font_scale (float): Font scale for text.
            font_thickness (int): Font thickness for text.
            font_color (tuple): Color of the text in (B, G, R) format.
            bg_color (tuple): Background color for the nutrition box in (B, G, R) format.
            transparency (float): Transparency level of the background (0.0 to 1.0).
        """
        font=cv2.FONT_HERSHEY_SIMPLEX; font_scale=0.40 ;font_thickness=1
        font_color=(255, 255, 255); bg_color=(0, 0, 0); transparency=0.5
        # Split nutrition data into lines
        nutrition = f'Nutritional Value for {class_name}:' + nutrition
        lines = nutrition.split("\n")
        
        # Position for the table (top-left corner)
        x_offset, y_offset = 10, 10
        line_height = 10  # Smaller line height for reduced size
        
        # Calculate dimensions of the background rectangle
        max_line_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines])
        table_height = line_height * len(lines)
        
        # Create a transparent overlay
        overlay = processed_img.copy()
        
        # Draw the background rectangle on the overlay
        cv2.rectangle(
            overlay,
            (x_offset, y_offset),
            (x_offset + max_line_width + 10, y_offset + table_height + 10),
            bg_color,
            -1,
        )
        
        # Blend the overlay with the original image for transparency
        cv2.addWeighted(overlay, transparency, processed_img, 1 - transparency, 0, processed_img)
        
        # Overlay each line of text on the image
        for i, line in enumerate(lines):
            y_line = y_offset + (i + 1) * line_height
            cv2.putText(processed_img, line, (x_offset + 5, y_line), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Save the processed image
        output_path = Path(output_directory) / image_name
        Path(output_directory).mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        cv2.imwrite(str(output_path), processed_img)
