import os
import sys
import subprocess
import time
import json

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("Checking requirements...")
    
    # Check if required Python packages are installed
    try:
        import pandas
        import numpy
        import joblib
        import sklearn
        print("[OK] Python dependencies installed")
    except ImportError as e:
        print(f"[ERROR] Missing Python dependency: {e}")
        print("Please install required packages: pip install pandas numpy scikit-learn joblib")
        return False
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Node.js installed: {result.stdout.strip()}")
        else:
            print("[ERROR] Node.js check failed")
            return False
    except FileNotFoundError:
        print("[ERROR] Node.js not found - please install Node.js")
        return False
    
    # Check if required files exist
    if not os.path.exists('sample_1000_accounts.csv'):
        print("[ERROR] Missing sample_1000_accounts.csv")
        return False
    
    if not os.path.exists('full_validation.js'):
        print("[ERROR] Missing full_validation.js")
        return False
    
    if not os.path.exists('bulk_account_validator.py'):
        print("[ERROR] Missing bulk_account_validator.py")
        return False
    
    # Create required directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("[OK] All requirements satisfied")
    return True

def ensure_model_exists():
    """Make sure the ML model exists"""
    model_path = 'account_validation_model.joblib'
    
    if os.path.exists(model_path):
        print(f"[OK] Model exists at {model_path}")
        return True
    
    print("Model not found, creating one...")
    try:
        # First check if create_account_model.py exists
        if os.path.exists('create_account_model.py'):
            result = subprocess.run(['python', 'create_account_model.py'], 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                print("[OK] Model created successfully")
                return True
            else:
                print(f"[ERROR] Error creating model: {result.stderr}")
                return False
        else:
            print("[ERROR] create_account_model.py not found")
            return False
    except Exception as e:
        print(f"[ERROR] Error creating model: {str(e)}")
        return False

def run_ml_prediction():
    """Run the ML prediction on accounts"""
    print("\n=== Running ML Prediction ===")
    try:
        # Set environment variable to disable plots in bulk_account_validator.py
        # This prevents GUI windows from appearing in a headless environment
        os.environ['SHOW_PLOTS'] = 'FALSE'
        
        result = subprocess.run(['python', 'bulk_account_validator.py'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print("[OK] ML prediction completed successfully")
            
            # Extract the prediction file from the output
            prediction_file = None
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Success: ML predictions saved to" in line:
                    prediction_file = line.replace("Success: ML predictions saved to ", "").strip()
                    break
            
            # Check if visualization files were created
            visualization_files = []
            for vis_file in ['static/prediction_distribution.png', 
                            'static/confidence_histogram.png', 
                            'static/predictions_by_bank.png']:
                if os.path.exists(vis_file):
                    visualization_files.append(vis_file)
            
            if visualization_files:
                print("\n[OK] Generated visualizations:")
                for vis_file in visualization_files:
                    print(f"  - {vis_file}")
                
                # On Windows, display the images using default viewer
                if sys.platform.startswith('win') and visualization_files:
                    show_visuals = input("\nDo you want to open the visualization images? (y/n): ").lower()
                    if show_visuals == 'y':
                        for vis_file in visualization_files:
                            try:
                                os.startfile(vis_file)
                                time.sleep(0.5)  # Small delay between opening files
                            except Exception as e:
                                print(f"[ERROR] Could not open {vis_file}: {str(e)}")
            
            if prediction_file:
                print(f"[OK] Predictions saved to {prediction_file}")
                return prediction_file
            else:
                print("[ERROR] Could not find prediction file in output")
                print(result.stdout)
                return None
        else:
            print(f"[ERROR] Error running ML prediction: {result.stderr}")
            return None
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return None

def run_api_validation(prediction_file):
    """Run API validation using the ML predictions"""
    if not prediction_file:
        print("[ERROR] No prediction file provided")
        return False
    
    print(f"\n=== Running API Validation with {prediction_file} ===")
    try:
        result = subprocess.run(['node', 'full_validation.js', '--predictions', prediction_file], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print("[OK] API validation completed successfully")
            
            # Extract output files
            validation_file = None
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Detailed results written to" in line:
                    validation_file = line.split("Detailed results written to")[1].strip()
                    break
            
            # Extract ML model accuracy if available
            ml_accuracy = None
            for line in output_lines:
                if "ML prediction accuracy" in line:
                    ml_accuracy = line.split("ML prediction accuracy:")[1].strip()
                    break
            
            if ml_accuracy:
                print(f"[OK] ML prediction accuracy: {ml_accuracy}")
            
            if validation_file:
                print(f"[OK] Validation results saved to {validation_file}")
            
            return True
        else:
            print(f"[ERROR] Error running API validation: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return False

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n=== Launching Dashboard ===")
    try:
        # Run in a new terminal window
        if sys.platform.startswith('win'):
            # For Windows
            process = subprocess.Popen(['start', 'cmd', '/k', 'python', '-m', 'streamlit', 'run', 'app.py'], 
                                    shell=True)
        else:
            # For macOS/Linux
            process = subprocess.Popen(['python', '-m', 'streamlit', 'run', 'app.py'],
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True)
        
        print("[OK] Dashboard launched")
        print("Open your browser at http://localhost:8501 to view the dashboard")
        print("Press Ctrl+C to exit the dashboard server when done")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error launching dashboard: {str(e)}")
        return False

def main():
    """Main workflow function"""
    print("=== PesaLink Fast Account Validation System ===\n")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("Please fix the requirements and try again")
        return
    
    # Step 2: Ensure model exists
    if not ensure_model_exists():
        print("Could not create or find ML model")
        return
    
    # Step 3: Run ML prediction
    prediction_file = run_ml_prediction()
    if not prediction_file:
        print("ML prediction failed")
        return
    
    # Step 4: Run API validation (optional)
    run_api = input("\nDo you want to run API validation with these predictions? (y/n): ").lower()
    if run_api == 'y':
        if not run_api_validation(prediction_file):
            print("API validation failed")
    
    # Step 5: Launch dashboard (optional)
    launch_dashboard = input("\nDo you want to launch the dashboard? (y/n): ").lower()
    if launch_dashboard == 'y':
        run_dashboard()
        try:
            # Keep the script running while the dashboard is open
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
    
    print("\n=== Workflow Complete ===")

if __name__ == "__main__":
    main() 