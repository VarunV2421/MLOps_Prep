import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def generate_shap_explanations():
    """
    Loads the trained model, calculates SHAP values for a SAMPLE of the test set,
    and saves global and individual explanation plots.
    """
    print("--- Generating SHAP Explanations ---")
    
    try:
        model = joblib.load("artifacts/model.pkl")
        df = pd.read_csv("data/iris.csv")
        expected_features = model.feature_names_in_
        X = df[expected_features]
        y = df['species']
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return

    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Use a smaller sample of the test set for performance ---
    if len(X_test) > 2000:
        X_test_sample = X_test.sample(n=2000, random_state=42)
    else:
        X_test_sample = X_test
    
    print(f"Calculating SHAP values for a sample of {len(X_test_sample)} instances...")
    
    explainer = shap.TreeExplainer(model)
    # This returns an Explanation object with shape (num_samples, num_features, num_classes)
    shap_values_explanation = explainer(X_test_sample)
    
    # --- Global Summary Plot ---
    print("Generating global summary plot...")
    plt.figure()
    # For the summary plot, we can pass the full explanation object
    shap.summary_plot(shap_values_explanation, X_test_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/shap_summary.png")
    plt.close()
    print("Global summary plot saved.")

    # --- Force Plot for All Instances in the Sample ---
    print("Generating stacked force plot...")
    
    # FIXED: Explicitly select the SHAP values and base value for the positive class (1)
    # This resolves the IndexError by passing the correctly shaped arrays to the plot.
    p_all = shap.force_plot(
        base_value=explainer.expected_value[1], 
        shap_values=shap_values_explanation.values[:, :, 1],
        features=X_test_sample,
        matplotlib=False
    )
    shap.save_html("artifacts/shap_force_plot_all.html", p_all)
    print("Stacked force plot saved.")
    print("--------------------------------------\n")

if __name__ == "__main__":
    generate_shap_explanations()
