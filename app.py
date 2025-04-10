from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model_tracker_cycle.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Your HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        features = [
            float(request.form['CycleWithPeakorNot']),
            float(request.form['ReproductiveCategory']),
            float(request.form['EstimatedDayofOvulation']),
            float(request.form['LengthofLutealPhase']),
            float(request.form['FirstDayofHigh']),
            float(request.form['TotalNumberofHighDays']),
            float(request.form['TotalHighPostPeak']),
            float(request.form['TotalNumberofPeakDays']),
            float(request.form['LengthofMenses']),
            float(request.form['MensesScoreDayOne']),
            float(request.form['MensesScoreDayTwo']),
            float(request.form['MensesScoreDayThree']),
            float(request.form['MensesScoreDayFour']),
            float(request.form['MensesScoreDayFive']),
            float(request.form['TotalMensesScore']),
            float(request.form['NumberofDaysofIntercourse']),
            float(request.form['IntercourseInFertileWindow']),
            float(request.form['UnusualBleeding'])
        ]
        
        cycle_number = 1  # Default for new cycle
        total_days_of_fertility = features[5] + features[7]  # High + Peak days
        total_fertility_formula = total_days_of_fertility + 1  # Or any logic you use
        length_of_cycle = features[2] + features[3]  # Ovulation day + luteal phase

        # Full input in the correct order:
        full_input = [
            cycle_number,
            *features[:2],                 # CycleWithPeakorNot, ReproductiveCategory
            length_of_cycle,
            *features[2:8],                # EstimatedDayofOvulation to TotalNumberofPeakDays
            total_days_of_fertility,
            total_fertility_formula,
            features[8],                   # LengthofMenses
            *features[9:]                  # Menses scores to UnusualBleeding
        ]

        # Predict using the model
        prediction = model.predict([full_input])[0]
        return f'<h3>Predicted Cycle Length: {round(prediction, 2)} days</h3>'
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
