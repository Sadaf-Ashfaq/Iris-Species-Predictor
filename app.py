from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_model.pkl')

# Define the valid ranges for each feature
RANGES = {
    'sepal_length': {'min': 4.3, 'max': 7.9},
    'sepal_width': {'min': 2.0, 'max': 4.4},
    'petal_length': {'min': 1.0, 'max': 6.9},
    'petal_width': {'min': 0.1, 'max': 2.5}
}

# Species information
SPECIES_INFO = {
    'setosa': {
        'common_name': 'Setosa Iris',
        'description': 'The Setosa iris is characterized by its small petals and wide sepals. Native to Alaska and parts of Canada.',
        'color': '#9b59b6'
    },
    'versicolor': {
        'common_name': 'Versicolor Iris',
        'description': 'The Versicolor iris has medium-sized flowers and is commonly found in the eastern United States.',
        'color': '#3498db'
    },
    'virginica': {
        'common_name': 'Virginica Iris',
        'description': 'The Virginica iris features the largest flowers among the three species, with long petals and sepals.',
        'color': '#e74c3c'
    }
}

@app.route('/')
def home():
    return render_template('index.html', ranges=RANGES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        # Validate ranges
        if not (RANGES['sepal_length']['min'] <= sepal_length <= RANGES['sepal_length']['max']):
            return jsonify({'error': f"Sepal length must be between {RANGES['sepal_length']['min']} and {RANGES['sepal_length']['max']} cm"}), 400
        
        if not (RANGES['sepal_width']['min'] <= sepal_width <= RANGES['sepal_width']['max']):
            return jsonify({'error': f"Sepal width must be between {RANGES['sepal_width']['min']} and {RANGES['sepal_width']['max']} cm"}), 400
        
        if not (RANGES['petal_length']['min'] <= petal_length <= RANGES['petal_length']['max']):
            return jsonify({'error': f"Petal length must be between {RANGES['petal_length']['min']} and {RANGES['petal_length']['max']} cm"}), 400
        
        if not (RANGES['petal_width']['min'] <= petal_width <= RANGES['petal_width']['max']):
            return jsonify({'error': f"Petal width must be between {RANGES['petal_width']['min']} and {RANGES['petal_width']['max']} cm"}), 400
        
        # Make prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get species information
        species_info = SPECIES_INFO[prediction]
        
        # Create probability dictionary
        prob_dict = {}
        for species, prob in zip(model.classes_, probabilities):
            prob_dict[species] = round(float(prob) * 100, 2)
        
        return jsonify({
            'prediction': prediction,
            'common_name': species_info['common_name'],
            'description': species_info['description'],
            'color': species_info['color'],
            'confidence': round(float(max(probabilities)) * 100, 2),
            'probabilities': prob_dict
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)