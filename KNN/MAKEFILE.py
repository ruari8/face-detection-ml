import os
import re
import csv

# Define the directory containing the log files
log_dir = r"C:\Users\User\Desktop\CSC3067-2425-G26\KNN"
output_csv = os.path.join(log_dir, "KNN_Results_Summary.csv")

# Define regex patterns to extract the important details
patterns = {
    "Accuracy": r"Accuracy:\s+([\d.]+)%",
    "Precision": r"Precision:\s+([\d.]+)",
    "Recall": r"Recall \(Sensitivity\):\s+([\d.]+)",
    "Specificity": r"Specificity:\s+([\d.]+)",
    "F1 Score": r"F1 Score:\s+([\d.]+)",
    "AUC": r"AUC:\s+([\d.]+)"
}

# List to store results
results = []

# Iterate over all log files in the directory
for log_file in os.listdir(log_dir):
    if log_file.endswith("_Log.txt"):
        script_name = log_file.replace("_Log.txt", "")
        file_path = os.path.join(log_dir, log_file)
        
        # Read the content of the log file
        with open(file_path, "r") as file:
            content = file.read()
        
        # Extract details using regex
        script_results = {"Script": script_name}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            script_results[key] = float(match.group(1)) if match else None
        
        # Add the results to the list
        results.append(script_results)

# Write the results to a CSV file
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["Script", "Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "AUC"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Results summary has been written to {output_csv}")
