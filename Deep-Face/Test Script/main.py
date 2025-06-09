from test import load_face_database, save_detailed_results, evaluate_face_recognition, visualize_results


# Load the pre-built database
db_file = "face_db.pkl"
face_db = load_face_database(db_file)

if face_db is None:
    print("Could not load face database. Exiting.")
    exit()

# Path to the folder with test images
test_folder = "KNOWN_FACES"

# Set up parameters
params = {
    "model_name": "ArcFace",
    "detector_backend": "retinaface",
    "threshold": 0.6,  # Adjust based on your needs
    "distance_metric": "cosine",
    "confidence_threshold": 0.4  # Minimum confidence to consider a valid match
}

# Run the evaluation
metrics = evaluate_face_recognition(test_folder, face_db, **params)

# Save detailed results
output_file = "face_recognition_results.csv"
save_detailed_results(metrics, output_file)

# Visualize the results
output_folder = "face_recognition_evaluation"
visualize_results(metrics, output_folder)

# Print final summary
print("\n===== FINAL SUMMARY =====")
print(f"Overall Recognition Accuracy: {metrics['overall_accuracy']:.2%}")
print(f"Face Detection Rate: {metrics['detection_rate']:.2%}")
print(f"Identification Accuracy (when faces detected): {metrics['identification_accuracy']:.2%}")

# Print per-identity accuracy for a clearer view
true_labels = metrics['true_labels']
predicted_labels = metrics['predicted_labels']

# Calculate per-identity stats
identity_stats = {}
for true, pred in zip(true_labels, predicted_labels):
    if true not in identity_stats:
        identity_stats[true] = {'total': 0, 'correct': 0}

    identity_stats[true]['total'] += 1
    if true == pred:
        identity_stats[true]['correct'] += 1

# Print per-identity accuracy
print("\nPer-Identity Accuracy:")
print("-" * 50)
print(f"{'ID':<15} {'Accuracy':<10} {'Correct/Total':<15}")
print("-" * 50)

for identity, stats in identity_stats.items():
    accuracy = stats['correct'] / stats['total']
    print(f"{identity:<15} {accuracy:.2%}      {stats['correct']}/{stats['total']}")
