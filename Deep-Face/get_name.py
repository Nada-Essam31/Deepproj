def extract_names_from_results(results):
    if isinstance(results, str):
        return results  # It's an error message or "No matching faces found"

    if "identity" not in results.columns:
        return "No identity column found in results"

    try:
        # Extract names from the 'identity' dictionaries
        names = results["identity"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
        unique_names = names.dropna().unique()
        
        if len(unique_names) == 0:
            return "No matching names found"

        # Return names as comma-separated string
        return ", ".join(unique_names)

    except Exception as e:
        return f"Error extracting names: {str(e)}"
