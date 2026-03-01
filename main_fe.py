import io
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image
from food_detection_and_nutritional_value.modeling import LLAMA, YOLOPredict


# Initialize models once at startup
print("Loading models...")
model_path = "models/last (1) (1).pt"
detector = YOLOPredict.YOLOInference(model_path)
llama = LLAMA.llamaOutput()
print("Models loaded successfully.")

def main(image):
    # Process image
    result = detector.process_image(image)
    
    detections = result[1]
    response = {
        'detected_items': [],
        'nutrition_info': '',
        'annotated_image': '',
        'detections_raw': []
    }
    
    processed_img = result[0]  # This should be a numpy array from YOLO
    
    if detections:
        # Process detections
        all_nutrition = []
        unique_classes = set()
        
        for det in detections:
            class_name = det['class']
            print(f"s- {class_name}: {det['score']:.2f}")
            
            # Record detection for UI markers
            response['detections_raw'].append({
                'class': class_name,
                'score': float(det['score'])
            })
            
            # Only process nutrition info once per unique food type
            if class_name not in unique_classes:
                print(f"Getting Nutritional information for {class_name}")
                nutrition = llama.nutrition_from_yolo_pred_class(class_name)
                formatted_table = llama.parse_nutrition_data(nutrition)
                
                response['detected_items'].append(class_name)
                all_nutrition.append(f"Nutritional Value for {class_name}:\n{formatted_table}")
                unique_classes.add(class_name)
        
        response['nutrition_info'] = "\n\n".join(all_nutrition)
        
        # Convert numpy array to image and then to base64
        try:
             # Convert numpy array (BGR) to PIL Image (RGB)
            if isinstance(processed_img, np.ndarray):
                # OpenCV handles BGR, PIL handles RGB
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
            else:
                img_pil = processed_img
            # Resize image while maintaining aspect ratio
            # Use 640 as a target max dimension for better display quality
            # (Higher than 128 to look good in UI, even though model uses 128)
            img_pil.thumbnail((640, 640), Image.Resampling.LANCZOS)

            # Save the resized image locally (optional)
            img_pil.save("resized_annotated.jpg", format="JPEG", quality=95)

            # Create a byte stream
            img_byte_arr = io.BytesIO()

            # Save the resized image as JPEG to the byte stream
            img_pil.save(img_byte_arr, format='JPEG', quality=95)

            # Get the byte array and encode it to Base64
            img_byte_arr = img_byte_arr.getvalue()
            base64_str = base64.b64encode(img_byte_arr).decode('utf-8')

            response['annotated_image'] = base64_str
            # Add all supported labels so the frontend can help the user
            response['supported_foods'] = list(detector.model.names.values())
            
        except Exception as e:
            import traceback
            print(f"Error encoding image: {str(e)}")
            traceback.print_exc()
            response['annotated_image'] = ''
    
    if not response['detected_items']:
        response['nutrition_info'] = 'No foods being detected!! :('
        # Still return the image if it wasn't returned inside the detections block
        if not response['annotated_image']:
            try:
                if isinstance(processed_img, np.ndarray):
                    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                else:
                    img_pil = processed_img
                
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format='JPEG', quality=95)
                base64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                response['annotated_image'] = base64_str
            except:
                pass
    
    return response
            


    # # Print results
    # for image_name, detections in results.items():
    #     print(f"\nDetections for {image_name}:")
    #     # print(detections)
    #     if detections:
    #         df[image_name] = {}
    #         processed_img =  detections['image']
    #         for det in detections:
    #             nutrition= llama.nutrition_from_yolo_pred_class(det['class'])
    #             # print(nutrition)
    #             # print(f"- {det['class']}: {det['score']:.2f}")
    #             formatted_table = llama.parse_nutrition_data(nutrition)
    #             print(formatted_table)
    #             df[image_name][det['class']] = formatted_table
    #             llama.put_nutrition_and_save(processed_img, formatted_table,output_directory, image_name, det['class'])
    #         cv2.imshow("Processed Frame", processed_img)
    #         cv2.destroyAllWindows()
        
    # pd.DataFrame(df).T.to_csv('validation_output.csv')
