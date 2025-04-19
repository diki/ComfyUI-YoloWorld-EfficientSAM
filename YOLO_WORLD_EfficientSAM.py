# --- Imports: Keep the necessary additions ---
from typing import List, Dict, Any # Added Dict, Any
import folder_paths
import os
import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from inference.models import YOLOWorld
import base64 # Keep
import json   # Keep
from io import BytesIO # Keep
from PIL import Image  # Keep

# --- Original Utilities (assuming these are correct) ---
from .utils.efficient_sam import load, inference_with_boxes
from .utils.video import generate_file_name, calculate_end_frame_index, create_directory

current_directory = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Original Annotators ---
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# --- Original folder_paths setup ---
folder_paths.folder_names_and_paths["yolo_world"] = ([os.path.join(folder_paths.models_dir, "yolo_world")], folder_paths.supported_pt_extensions)

# --- Original process_categories ---
def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]

# --- Original annotate_image ---
def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
    thickness: int = 2,
    text_thickness: int = 2,
    text_scale: float = 1.0,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        # Add basic check to prevent index out of bounds if detections/categories mismatch
        for class_id, confidence in zip(detections.class_id, detections.confidence) if class_id < len(categories)
    ]
    # Filter detections corresponding to valid labels
    valid_indices = [i for i, class_id in enumerate(detections.class_id) if class_id < len(categories)]
    valid_detections = detections[valid_indices]

    if len(valid_detections) == 0:
        return input_image # Return original if no valid detections

    # Use local instances of annotators to respect thickness/scale parameters for this call
    local_bbox_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
    local_mask_annotator = sv.MaskAnnotator()
    local_label_annotator = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)

    output_image = local_mask_annotator.annotate(input_image.copy(), valid_detections)
    output_image = local_bbox_annotator.annotate(output_image, valid_detections)
    output_image = local_label_annotator.annotate(output_image, valid_detections, labels=labels)
    return output_image


# --- Yoloworld_ModelLoader_Zho: REVERTED TO ORIGINAL ---
class Yoloworld_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_world_model": (["yolo_world/l", "yolo_world/m", "yolo_world/s"], ),
            }
        }

    RETURN_TYPES = ("YOLOWORLDMODEL",)
    RETURN_NAMES = ("yolo_world_model",)
    FUNCTION = "load_yolo_world_model"
    CATEGORY = "🔎YOLOWORLD_ESAM"

    def load_yolo_world_model(self, yolo_world_model):
        # --- EXACT ORIGINAL LOADING LOGIC ---
        YOLO_WORLD_MODEL = YOLOWorld(model_id=yolo_world_model)
        return [YOLO_WORLD_MODEL]


# --- ESAM_ModelLoader_Zho: REVERTED TO ORIGINAL ---
class ESAM_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["CUDA", "CPU"], ),
            }
        }

    RETURN_TYPES = ("ESAMMODEL",)
    RETURN_NAMES = ("esam_model",)
    FUNCTION = "load_esam_model"
    CATEGORY = "🔎YOLOWORLD_ESAM"

    def load_esam_model(self, device):
        # --- EXACT ORIGINAL LOADING LOGIC ---
        if device == "CUDA":
            model_path = os.path.join(current_directory, "efficient_sam_s_gpu.jit")
        else:
            model_path = os.path.join(current_directory, "efficient_sam_s_cpu.jit")
        # Basic check if file exists before loading
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"ESAM model not found at expected path: {model_path}")

        EFFICIENT_SAM_MODEL = torch.jit.load(model_path)
        # Ensure model is on correct device after loading
        target_device = DEVICE # Use the globally defined DEVICE
        EFFICIENT_SAM_MODEL.to(target_device)
        print(f"Loaded ESAM model from {model_path} to {target_device}")

        return [EFFICIENT_SAM_MODEL]

# --- Yoloworld_ESAM_Zho: Keep JSON output additions, Revert INPUT_TYPES ---
class Yoloworld_ESAM_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # --- REVERTED TO ORIGINAL INPUT TYPES ---
        return {
            "required": {
                "yolo_world_model": ("YOLOWORLDMODEL",),
                "esam_model": ("ESAMMODEL",),
                "image": ("IMAGE",),
                "categories": ("STRING", {"default": "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step":0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step":0.01}),
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step":0.1}), # Keep adjusted range from previous attempt
                "with_confidence": ("BOOLEAN", {"default": True}),
                "with_class_agnostic_nms": ("BOOLEAN", {"default": False}),
                "with_segmentation": ("BOOLEAN", {"default": True}),
                "mask_combined": ("BOOLEAN", {"default": True}), # Original boolean flag
                "mask_extracted": ("BOOLEAN", {"default": True}), # Original boolean flag
                "mask_extracted_index": ("INT", {"default": 0, "min": 0, "max": 1000}), # Original index
            }
        }

    # --- Updated Return Types/Names (Keep this change) ---
    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("image", "mask", "label_mask_data",)
    FUNCTION = "yoloworld_esam_image"
    CATEGORY = "🔎YOLOWORLD_ESAM"

    # --- Helper function (Keep this) ---
    def encode_mask_to_base64(self, mask_np: np.ndarray) -> str:
        """Converts a numpy mask (HxW, bool or uint8) to a base64 PNG string."""
        if mask_np.dtype == bool:
             mask_uint8 = (mask_np * 255).astype(np.uint8)
        elif mask_np.dtype == np.uint8:
             mask_uint8 = mask_np
        else:
             mask_uint8 = (mask_np > 0).astype(np.uint8) * 255

        try:
            pil_mask = Image.fromarray(mask_uint8, mode='L')
            buffer = BytesIO()
            pil_mask.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
            base64_string = base64.b64encode(png_bytes).decode('utf-8')
            return base64_string
        except Exception as e:
            print(f"Error encoding mask to base64: {e}")
            return ""

    # --- Main function: Integrate JSON generation into original structure ---
    def yoloworld_esam_image(self, image, yolo_world_model, esam_model, categories, confidence_threshold, iou_threshold, box_thickness, text_thickness, text_scale, with_segmentation, mask_combined, with_confidence, with_class_agnostic_nms, mask_extracted, mask_extracted_index):
        categories = process_categories(categories)
        processed_images = []
        processed_masks = []
        label_mask_data_batch = [] # To store structured data

        for img in image: # Original loop structure
            img_np_rgb = np.clip(255. * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            YOLO_WORLD_MODEL = yolo_world_model
            YOLO_WORLD_MODEL.set_classes(categories)
            results = YOLO_WORLD_MODEL.infer(img_np_rgb, confidence=confidence_threshold)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )

            current_image_label_mask_data = [] # Data for this specific image
            combined_mask_tensor_for_output = torch.empty(0) # Mask tensor to be returned

            if with_segmentation and len(detections) > 0:
                try:
                    detections.mask = inference_with_boxes(
                        image=img_np_rgb,
                        xyxy=detections.xyxy,
                        model=esam_model,
                        device=DEVICE # Use global DEVICE
                    )

                    # --- Generate Label/Mask Data (Keep this logic) ---
                    if detections.mask is not None:
                        num_detections = len(detections)
                        num_masks = len(detections.mask)
                        if num_detections == num_masks:
                             valid_mask_indices = []
                             for i in range(num_detections):
                                 if detections.class_id[i] < len(categories):
                                     label = categories[detections.class_id[i]]
                                     individual_mask_np = detections.mask[i]
                                     mask_base64 = self.encode_mask_to_base64(individual_mask_np)
                                     if mask_base64:
                                         current_image_label_mask_data.append({
                                             "label": label,
                                             "mask_base64": mask_base64,
                                             "confidence": float(detections.confidence[i])
                                         })
                                         valid_mask_indices.append(i)
                                     else:
                                         print(f"Warning: Failed to encode mask for detection {i} ({label}).")
                                 else:
                                     print(f"Warning: class_id {detections.class_id[i]} out of bounds for categories (len {len(categories)}).")
                             # Filter detections to match successfully processed data
                             detections = detections[valid_mask_indices]
                        else:
                             print(f"Warning: Mismatch detections ({num_detections}) vs masks ({num_masks}). Skipping label/mask data generation.")
                             detections.mask = None # Invalidate masks if counts don't match

                    # --- Original Mask Tensor Output Logic ---
                    if detections.mask is not None:
                        if mask_combined:
                            combined_mask_np = np.logical_or.reduce(detections.mask, axis=0)
                            combined_mask_tensor_for_output = torch.tensor(combined_mask_np.astype(np.uint8), dtype=torch.float32) # Use uint8 then cast
                        else:
                            det_mask_np = detections.mask # Shape (N, H, W)
                            if mask_extracted:
                                if 0 <= mask_extracted_index < len(det_mask_np):
                                     selected_mask_np = det_mask_np[mask_extracted_index]
                                     combined_mask_tensor_for_output = torch.tensor(selected_mask_np.astype(np.uint8), dtype=torch.float32)
                                else:
                                     print(f"Warning: mask_extracted_index {mask_extracted_index} out of bounds for {len(det_mask_np)} masks.")
                                     # combined_mask_tensor_for_output remains empty
                            else:
                                # Output all individual masks as a batch tensor
                                combined_mask_tensor_for_output = torch.tensor(det_mask_np.astype(np.uint8), dtype=torch.float32) # Shape (N, H, W)

                except Exception as e:
                    print(f"Error during segmentation or mask processing: {e}")
                    detections.mask = None # Ensure masks are None if error occurred
                    current_image_label_mask_data = [] # Clear data on error

            else: # No segmentation or no detections
                 detections.mask = None # Ensure masks are None

            processed_masks.append(combined_mask_tensor_for_output) # Append the tensor for this image (might be empty)
            label_mask_data_batch.append(current_image_label_mask_data) # Append data for this image

            # --- Original Image Annotation Logic ---
            output_image_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

            # Annotate using potentially filtered detections
            output_image_annotated = annotate_image(
                input_image=output_image_bgr,
                detections=detections, # Use detections potentially filtered above
                categories=categories,
                with_confidence=with_confidence,
                thickness=box_thickness,
                text_thickness=text_thickness,
                text_scale=text_scale,
            )

            output_image_rgb = cv2.cvtColor(output_image_annotated, cv2.COLOR_BGR2RGB)
            output_image_tensor = torch.from_numpy(output_image_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            processed_images.append(output_image_tensor)


        # --- Combine Batch Results (Original Logic for Image/Mask) ---
        new_ims = torch.cat(processed_images, dim=0)

        # Handle masks - output only the first image's mask tensor for simplicity
        # (Same logic as previous attempt to ensure single tensor output)
        if processed_masks:
             final_masks_tensor = processed_masks[0] # Take mask tensor from first image
        else:
             final_masks_tensor = torch.empty(0)


        # --- Format Structured Data Output (Keep This) ---
        first_image_data = label_mask_data_batch[0] if label_mask_data_batch else []
        json_output_string = json.dumps(first_image_data, indent=2)

        # --- Return: Original image/mask + new JSON string ---
        return new_ims, final_masks_tensor, json_output_string


# --- Mappings: Keep as is ---
NODE_CLASS_MAPPINGS = {
    "Yoloworld_ModelLoader_Zho": Yoloworld_ModelLoader_Zho,
    "ESAM_ModelLoader_Zho": ESAM_ModelLoader_Zho,
    "Yoloworld_ESAM_Zho": Yoloworld_ESAM_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yoloworld_ModelLoader_Zho": "🔎Yoloworld Model Loader",
    "ESAM_ModelLoader_Zho": "🔎ESAM Model Loader",
    "Yoloworld_ESAM_Zho": "🔎Yoloworld ESAM",
}