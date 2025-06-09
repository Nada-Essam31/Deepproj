
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def display_results(img_path, results, confidence_threshold=0.4):

    if isinstance(results, str):
        print(results)
        return
    
    if len(results) == 0:
        print("No results to display")
        return
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    ax = plt.gca()
    
    # Process each detected face
    processed_regions = set()
    
    for _, row in results.iterrows():
        # Get the face region
        region_key = f"{row['source_x']}-{row['source_y']}-{row['source_w']}-{row['source_h']}"
        
        # Skip if we've already processed this region
        if region_key in processed_regions:
            continue
        
        processed_regions.add(region_key)
        
        # Get the identity information
        identity = row['identity']
        if isinstance(identity, dict):
            identity_name = identity.get('name', 'Unknown')
        else:
            # Fall back to extracting from path if needed
            try:
                identity_name = os.path.basename(os.path.dirname(identity))
            except:
                identity_name = "Unknown"
        
        # Calculate confidence
        confidence = 1 - row['distance']
        
        # Only show matches above the threshold
        if confidence < confidence_threshold:
            identity_name = "Unknown"
        
        # Draw a rectangle around the face
        rect = Rectangle(
            (row['source_x'], row['source_y']),
            row['source_w'], row['source_h'],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add a label with the identity
        label = f"{identity_name} ({confidence:.2f})"
        plt.text(
            row['source_x'], row['source_y'] - 10,
            label,
            color='white', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='red', alpha=0.7)
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

