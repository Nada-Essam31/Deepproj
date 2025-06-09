from deepface import DeepFace
import os
import pandas as pd
import numpy as np
import pickle
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def build_face_database(db_path, output_path=None, model_name="ArcFace", detector_backend="retinaface"):

    print("Building face database...")
    start_time = time.time()
    

    image_paths = []
    for root, dirs, files in os.walk(db_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images in database directory.")
    if len(image_paths) == 0:
        print("No images found in the database path.")
        return None
    
    # Extract face embeddings for all images
    embeddings = []
    identities = []
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"Processing image {i}/{len(image_paths)}: {img_path}")
        try:
            # Extract embedding
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend
            )
            
            # If face found, add to database
            if embedding_objs:
                # Get identity name from parent folder name
                identity_name = os.path.basename(os.path.dirname(img_path))
                print(f"  Found {len(embedding_objs)} face(s) for identity: {identity_name}")
                
                for embedding_obj in embedding_objs:
                    embeddings.append(embedding_obj["embedding"])
                    identities.append({"name": identity_name})
            else:
                print(f"  No faces detected in {img_path}")
                    
        except Exception as e:
            print(f"  Error processing {img_path}: {str(e)}")
    
    # Create a DataFrame to store the embeddings
    if len(embeddings) == 0:
        print("No faces detected in any database images.")
        return None
    
    df = pd.DataFrame({
        "identity": identities,
        "embedding": embeddings
    })
    
    # Save the database
    if output_path is None:
        output_path = os.path.dirname(db_path)
    
    db_file = os.path.join(output_path, "face_db.pkl")
    with open(db_file, "wb") as f:
        pickle.dump(df, f)
    
    print(f"Database built successfully with {len(df)} faces in {time.time() - start_time:.2f} seconds.")
    print(f"Database saved to {db_file}")
    
    return df

def load_face_database(db_file):

    if os.path.isfile(db_file):
        print(f"Loading face database from {db_file}...")
        with open(db_file, "rb") as f:
            face_db = pickle.load(f)
        print(f"Loaded database with {len(face_db)} face entries.")
        return face_db
    else:
        print(f"Database file {db_file} not found.")
        return None

