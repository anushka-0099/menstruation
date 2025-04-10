from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = 'model_tracker_cycle.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Field mapping with conversion functions and defaults
        field_mapping = {
            'CycleWithPeakorNot': ('CycleWithPeakorNot', lambda x: 1 if x == '1' else 0, 0),
            'ReproductiveCategory': ('ReproductiveCategory', float, 1),
            'LengthofLutealPhase': ('LengthofLutealPhase', float, 14),
            'FirstDayofHigh': ('FirstDayofHigh', float, 10),
            'TotalNumberofHighDays': ('TotalNumberofHighDays', float, 5),
            'TotalHighPostPeak': ('TotalHighPostPeak', float, 2),
            'TotalNumberofPeakDays': ('TotalNumberofPeakDays', float, 0),
            'LengthofMenses': ('LengthofMenses', float, 5),
            'MensesScoreDayOne': ('MensesScoreDayOne', float, 2),
            'MensesScoreDayTwo': ('MensesScoreDayTwo', float, 2),
            'MensesScoreDayThree': ('MensesScoreDayThree', float, 1),
            'MensesScoreDayFour': ('MensesScoreDayFour', float, 1),
            'MensesScoreDayFive': ('MensesScoreDayFive', float, 0),
            'NumberofDaysofIntercourse': ('NumberofDaysofIntercourse', float, 0),
            'IntercourseInFertileWindow': ('IntercourseInFertileWindow', float, 0),
            'UnusualBleeding': ('UnusualBleeding', float, 0)
        }
        
        # Process form data
        processed_data = {}
        for field, (form_field, converter, default) in field_mapping.items():
            value = request.form.get(form_field)
            processed_data[field] = converter(value) if value is not None else default
        
        # Calculate derived fields
        processed_data['EstimatedDayofOvulation'] = 28 - processed_data['LengthofLutealPhase']
        processed_data['TotalMensesScore'] = sum([
            processed_data['MensesScoreDayOne'],
            processed_data['MensesScoreDayTwo'],
            processed_data['MensesScoreDayThree'],
            processed_data['MensesScoreDayFour'],
            processed_data['MensesScoreDayFive']
        ])
        
        # Calculate fertility metrics
        total_days_of_fertility = processed_data['TotalNumberofHighDays'] + processed_data['TotalNumberofPeakDays']
        total_fertility_formula = total_days_of_fertility + 1
        length_of_cycle = processed_data['EstimatedDayofOvulation'] + processed_data['LengthofLutealPhase']
        
        # Prepare full feature vector
        features = [
            1,  # CycleNumber
            processed_data['CycleWithPeakorNot'],
            processed_data['ReproductiveCategory'],
            length_of_cycle,
            processed_data['EstimatedDayofOvulation'],
            processed_data['LengthofLutealPhase'],
            processed_data['FirstDayofHigh'],
            processed_data['TotalNumberofHighDays'],
            processed_data['TotalHighPostPeak'],
            processed_data['TotalNumberofPeakDays'],
            total_days_of_fertility,
            total_fertility_formula,
            processed_data['LengthofMenses'],
            processed_data['MensesScoreDayOne'],
            processed_data['MensesScoreDayTwo'],
            processed_data['MensesScoreDayThree'],
            processed_data['MensesScoreDayFour'],
            processed_data['MensesScoreDayFive'],
            processed_data['TotalMensesScore'],
            processed_data['NumberofDaysofIntercourse'],
            processed_data['IntercourseInFertileWindow'],
            processed_data['UnusualBleeding']
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        return render_template('result.html',
                            prediction=round(length_of_cycle, 2))
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)