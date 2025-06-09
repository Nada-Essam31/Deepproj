from deepface import DeepFace


import pandas as pd
import numpy as np


def find_faces(img_path, face_db, model_name="ArcFace", 
               detector_backend="retinaface", distance_metric="cosine", threshold=0.6):

    if face_db is None or len(face_db) == 0:
        return "No faces in database"
    
    print(f"Processing query image: {img_path}")
    
    # Extract embedding for the query image
    try:
        target_embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend
        )
        print(f"Detected {len(target_embedding_objs)} faces in query image.")
    except Exception as e:
        error_msg = f"Error processing query image: {str(e)}"
        print(error_msg)
        return error_msg
    
    if not target_embedding_objs:
        return "No face detected in the query image"
    
    # Compare the query embedding with all embeddings in the database
    results = []
    
    for i, target_embedding_obj in enumerate(target_embedding_objs):
        print(f"Processing face {i+1}/{len(target_embedding_objs)} in query image")
        target_embedding = target_embedding_obj["embedding"]
        
        distances = []
        # Get the database embeddings
        db_embeddings = list(face_db["embedding"])
        
        print(f"Comparing with {len(db_embeddings)} faces in database...")
        for db_embedding in db_embeddings:
            if distance_metric == "cosine":
                distance = 1 - np.dot(db_embedding, target_embedding) / (
                    np.linalg.norm(db_embedding) * np.linalg.norm(target_embedding)
                )
            elif distance_metric == "euclidean":
                distance = np.linalg.norm(np.array(db_embedding) - np.array(target_embedding))
            else:
                raise ValueError(f"Distance metric {distance_metric} not supported")
            
            distances.append(distance)
        
        # Create a result DataFrame
        result_df = face_db.copy()
        result_df["distance"] = distances
        
        # Filter by threshold
        result_df = result_df[result_df["distance"] <= threshold]
        
        # Sort by distance
        result_df = result_df.sort_values(by=["distance"])
        
        # Add the region information
        region = target_embedding_obj.get("facial_area", {})
        result_df["source_x"] = region.get("x", 0)
        result_df["source_y"] = region.get("y", 0) 
        result_df["source_w"] = region.get("w", 0)
        result_df["source_h"] = region.get("h", 0)
        
        print(f"Found {len(result_df)} matches for face {i+1}")
        
        results.append(result_df)
    
    # Combine all results
    if len(results) > 0:
        combined_result = pd.concat(results)
        return combined_result
    else:
        return "No matching faces found"