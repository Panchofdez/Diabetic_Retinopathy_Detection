# Diabetic_Retinopathy_Detection

This project is designed to classify the severity of diabetic retinopathy

## Steps to Run Locally

1. **Unzip Dataset**  
   Extract the dataset from the `dataset.zip` file:

   ```
   unzip dataset.zip
   ```

2. **Unzip Pre-trained Model**  
   Extract the pre-trained model from the dr_classification_model.zip file:

   ```
   bash unzip dr_classification_model.zip
   ```

3. **Create a Virtual Environment**  
   Create and activate a Python virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

4. **Install Required Dependencies**

   ```
   pip install -r requirements.txt
   ```

5. **Run the Script**  
   _Note_: If you want to train the model then uncomment the train_model function found at the bottom of the script
   ```
   python dr_detection.py
   ```

- A kaggle notebook that was used was also included if that is preferred
