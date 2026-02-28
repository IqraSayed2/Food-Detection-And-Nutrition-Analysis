import os
from food_detection_and_nutritional_value.modeling import YOLOPredict, LLAMA
from PIL import Image

# Set paths
input_folder = 'uploads/raw'
output_folder = 'batch_processed_results'
model_path = 'models/last (1) (1).pt'

# Initialize detector and nutrition model
detector = YOLOPredict.YOLOInference(model_path)
llama = LLAMA.llamaOutput()

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(input_folder, filename)
        try:
            image = Image.open(image_path)
            processed_img, detections = detector.process_image(image)
            print(f"{filename}: {len(detections)} detections")
            for det in detections:
                print(f"  - {det['class']} (score: {det['score']:.2f})")
                nutrition = llama.nutrition_from_yolo_pred_class(det['class'])
                formatted_table = llama.parse_nutrition_data(nutrition)
                print(formatted_table)
                # Optionally, save annotated image with nutrition info overlay
                llama.put_nutrition_and_save(processed_img, formatted_table, output_folder, filename, det['class'])
        except Exception as e:
            print(f"Error processing {filename}: {e}")
