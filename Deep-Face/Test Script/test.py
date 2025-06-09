import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from deepface import DeepFace
from build_face_database import load_face_database       
def evaluate_face_recognition(test_folder, face_db, model_name="ArcFace", 
                           detector_backend="retinaface", threshold=0.6, 
                           distance_metric="cosine", confidence_threshold=0.4):
    """
    Evaluate face recognition accuracy on a folder of test images
    
    Parameters:
    -----------
    test_folder : str
        Path to the folder containing test face images
    face_db : pd.DataFrame
        Pre-built face database
    model_name : str, default "ArcFace"
        The face recognition model to use
    detector_backend : str, default "retinaface"
        The face detection model to use
    threshold : float, default 0.6
        The threshold for face matching
    distance_metric : str, default "cosine"
        The distance metric to use for face comparison
    confidence_threshold : float, default 0.4
        Confidence threshold for a match to be considered valid
        
    Returns:
    --------
    dict
        Dictionary containing various accuracy metrics
    """
    print(f"Evaluating face recognition accuracy on folder: {test_folder}")
    
    # Get all image paths in the test folder
    image_paths = []
    true_labels = []
    
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                
                # Extract true label from the filename without extension
                # For example, "C2202210.jpeg" becomes "C2202210"
                filename = os.path.basename(image_path)
                true_label = os.path.splitext(filename)[0]  # Remove file extension
                true_labels.append(true_label)
    
    print(f"Found {len(image_paths)} test images")
    
    # Initialize results storage
    predicted_labels = []
    confidence_scores = []
    detection_success = []
    
    # Process each test image
    for i, (img_path, true_label) in enumerate(zip(image_paths, true_labels), 1):
        print(f"Processing test image {i}/{len(image_paths)}: {img_path}")
        print(f"True label: {true_label}")
        
        # Run face recognition
        try:
            results = find_faces(
                img_path=img_path,
                face_db=face_db,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                threshold=threshold
            )
            
            if isinstance(results, str):
                print(f"  Error: {results}")
                predicted_labels.append("unknown")
                confidence_scores.append(0.0)
                detection_success.append(False)
                continue
                
            if len(results) == 0:
                print("  No faces detected")
                predicted_labels.append("unknown")
                confidence_scores.append(0.0)
                detection_success.append(False)
                continue
            
            # Get the best match (lowest distance)
            best_match = results.iloc[0]
            distance = best_match['distance']
            confidence = 1 - distance
            
            # Extract just the name from the identity column
            if isinstance(best_match['identity'], dict):
                predicted_label = best_match['identity'].get('name', 'unknown')
            else:
                # Fallback
                try:
                    predicted_label = os.path.basename(os.path.dirname(best_match['identity']))
                except:
                    predicted_label = "unknown"
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                predicted_label = "unknown"
                
            print(f"  Predicted: {predicted_label} with confidence: {confidence:.4f}")
            
            predicted_labels.append(predicted_label)
            confidence_scores.append(confidence)
            detection_success.append(True)
            
        except Exception as e:
            print(f"  Error processing {img_path}: {str(e)}")
            predicted_labels.append("unknown")
            confidence_scores.append(0.0)
            detection_success.append(False)
    
    # Calculate accuracy metrics
    detected_indices = [i for i, success in enumerate(detection_success) if success]
    
    if len(detected_indices) == 0:
        print("No faces were successfully detected in any test images")
        return {
            "overall_accuracy": 0.0,
            "detection_rate": 0.0,
            "identification_accuracy": 0.0,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "confidence_scores": confidence_scores
        }
    
    # Detection rate (percentage of images where a face was detected)
    detection_rate = len(detected_indices) / len(image_paths)
    
    # Overall accuracy (correct predictions / total images)
    correct_predictions = sum(1 for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)) 
                             if true == pred)
    overall_accuracy = correct_predictions / len(image_paths)
    
    # Identification accuracy (correct predictions / images where faces were detected)
    detected_true_labels = [true_labels[i] for i in detected_indices]
    detected_predicted_labels = [predicted_labels[i] for i in detected_indices]
    identification_accuracy = sum(1 for true, pred in zip(detected_true_labels, detected_predicted_labels) 
                                 if true == pred) / len(detected_indices)
    
    # Print summary
    print("\n----- Accuracy Summary -----")
    print(f"Total test images: {len(image_paths)}")
    print(f"Successfully detected faces: {len(detected_indices)} ({detection_rate:.2%})")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print(f"Identification accuracy: {identification_accuracy:.2%}")
    
    # Create classification report for detected faces
    if len(set(detected_true_labels)) > 1:
        print("\nClassification Report (for detected faces):")
        print(classification_report(detected_true_labels, detected_predicted_labels))
    
    # Return all metrics
    return {
        "overall_accuracy": overall_accuracy,
        "detection_rate": detection_rate,
        "identification_accuracy": identification_accuracy,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "confidence_scores": confidence_scores
    }

def visualize_results(metrics, output_folder=None):
    """
    Visualize the evaluation results
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing accuracy metrics
    output_folder : str, optional
        Folder to save the visualizations
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extract data
    true_labels = metrics['true_labels']
    predicted_labels = metrics['predicted_labels']
    
    # Only include results where faces were detected
    detected_indices = [i for i, conf in enumerate(metrics['confidence_scores']) if conf > 0]
    
    if len(detected_indices) == 0:
        print("No faces were detected, cannot create visualizations")
        return
    
    detected_true = [true_labels[i] for i in detected_indices]
    detected_pred = [predicted_labels[i] for i in detected_indices]
    
    # Get unique labels (excluding "unknown")
    all_labels = set(detected_true + detected_pred)
    if "unknown" in all_labels:
        all_labels.remove("unknown")
    all_labels = sorted(list(all_labels))
    
    # Create confusion matrix
    if len(all_labels) > 1:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(detected_true, detected_pred, labels=all_labels)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if output_folder:
            plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
        plt.show()
    
    # Create accuracy summary bar chart
    plt.figure(figsize=(10, 6))
    metrics_to_plot = [
        ('Detection Rate', metrics['detection_rate']),
        ('Overall Accuracy', metrics['overall_accuracy']),
        ('Identification Accuracy', metrics['identification_accuracy'])
    ]
    
    plt.bar([x[0] for x in metrics_to_plot], [x[1] for x in metrics_to_plot])
    plt.ylabel('Score')
    plt.title('Face Recognition Performance Metrics')
    plt.ylim(0, 1.0)
    
    # Add percentage labels on bars
    for i, (label, value) in enumerate(metrics_to_plot):
        plt.text(i, value + 0.01, f'{value:.1%}', ha='center')
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, "accuracy_summary.png"))
    plt.show()
    
    # Create per-identity accuracy chart if enough data
    identity_counts = {}
    identity_correct = {}
    
    for true, pred in zip(true_labels, predicted_labels):
        if true not in identity_counts:
            identity_counts[true] = 0
            identity_correct[true] = 0
        
        identity_counts[true] += 1
        if true == pred:
            identity_correct[true] += 1
    
    # Sort identities by count for better visualization
    identities = sorted(identity_counts.keys(), key=lambda x: identity_counts[x], reverse=True)
    
    if len(identities) > 1:
        plt.figure(figsize=(12, 6))
        accuracy_per_identity = [identity_correct[identity] / identity_counts[identity] 
                                for identity in identities]
        
        plt.bar(identities, accuracy_per_identity)
        plt.xlabel('Identity')
        plt.ylabel('Accuracy')
        plt.title('Recognition Accuracy per Identity')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        
        # Add sample count on top of bars
        for i, identity in enumerate(identities):
            plt.text(i, accuracy_per_identity[i] + 0.01, 
                     f'{identity_correct[identity]}/{identity_counts[identity]}', 
                     ha='center')
        
        plt.tight_layout()
        if output_folder:
            plt.savefig(os.path.join(output_folder, "accuracy_per_identity.png"))
        plt.show()
    
    # Create confidence distribution plot
    plt.figure(figsize=(10, 6))
    detected_conf = [metrics['confidence_scores'][i] for i in detected_indices]
    correct_indices = [i for i, (true, pred) in enumerate(zip(detected_true, detected_pred)) 
                      if true == pred]
    incorrect_indices = [i for i, (true, pred) in enumerate(zip(detected_true, detected_pred)) 
                        if true != pred]
    
    if correct_indices:
        correct_conf = [detected_conf[i] for i in correct_indices]
        plt.hist(correct_conf, alpha=0.5, label='Correct Predictions', bins=20, range=(0, 1))
    
    if incorrect_indices:
        incorrect_conf = [detected_conf[i] for i in incorrect_indices]
        plt.hist(incorrect_conf, alpha=0.5, label='Incorrect Predictions', bins=20, range=(0, 1))
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, "confidence_distribution.png"))
    plt.show()

def save_detailed_results(metrics, output_file):
    """
    Save detailed results to a CSV file
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing accuracy metrics
    output_file : str
        Path to save the CSV file
    """
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'true_label': metrics['true_labels'],
        'predicted_label': metrics['predicted_labels'],
        'confidence': metrics['confidence_scores']
    })
    
    # Add a column for correct/incorrect
    results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")

# Modify find_faces function to extract just the name from identity
def find_faces(img_path, face_db, model_name="ArcFace", 
               detector_backend="retinaface", distance_metric="cosine", threshold=0.6):
    """
    Search for faces in the pre-built database
    
    Parameters:
    -----------
    img_path : str
        Path to the query image
    face_db : pd.DataFrame
        Pre-built face database
    model_name : str, default "ArcFace"
        The face recognition model to use
    detector_backend : str, default "retinaface"
        The face detection model to use
    distance_metric : str, default "cosine"
        The distance metric to use for face comparison
    threshold : float, default 0.6
        The threshold for face matching
        
    Returns:
    --------
    pd.DataFrame or str
        DataFrame containing matching results or error message
    """
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
        
        # Extract just the name from identity
        def extract_name(identity_dict):
            if isinstance(identity_dict, dict):
                return identity_dict.get('name', 'Unknown')
            return 'Unknown'
        
        # Apply the name extraction to all rows
        if len(result_df) > 0:
            result_df['name'] = result_df['identity'].apply(extract_name)
        
        print(f"Found {len(result_df)} matches for face {i+1}")
        
        results.append(result_df)
    
    # Combine all results
    if len(results) > 0:
        combined_result = pd.concat(results)
        return combined_result
    else:
        return "No matching faces found"

