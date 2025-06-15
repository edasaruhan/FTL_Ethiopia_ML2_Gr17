from flask import Flask, request, render_template, redirect, url_for, flash, get_flashed_messages
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = '19f76b6a66e6f742be3e66027b11e9ba'  # secret key used for the flash messages

# Load the full pipeline (scaler + model)
try:
    pipeline = joblib.load('optimized_water_access_model.joblib')
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input values
            # Change the number into whole number
            gdp_log = float(request.form['gdp_log'])
            pop_density = float(request.form['pop_density'])
            urban_pop = float(request.form['urban_pop'])
            sanitation = float(request.form['sanitation'])

            # Prepare input to make sure they got the same order as the training
            input_data = np.array([[gdp_log, pop_density, urban_pop, sanitation]])

            # Use full pipeline for prediction (automatically scales input)
            if pipeline:
                prediction = pipeline.predict(input_data)[0]
                flash(f"{prediction:.2f}")
            else:
                flash("Model failed to load")
        except ValueError:
            flash("Invalid input. Please enter numbers only.")
        except Exception as e:
            flash(f"Error: {str(e)}")
        return redirect(url_for('index'))

    # Get prediction from flash message
    messages = get_flashed_messages()
    prediction = messages[0] if messages else None
    return render_template('index.html', prediction=prediction)

@app.after_request
def add_header(response):
    """Prevent caching of dynamic content"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)





