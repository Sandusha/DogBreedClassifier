from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('dogbreedid.h5')

# Define the dog breed classes directly within the app
dog_breeds = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier',
    'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier',
    'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel',
    'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber',
    'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman',
    'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
    'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever',
    'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound',
    'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel',
    'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
    'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher',
    'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug',
    'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
    'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky',
    'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle',
    'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
    'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
    'wire-haired_fox_terrier', 'yorkshire_terrier'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded file to a temporary location
            img_path = 'static/uploads/temp_image.jpg'  # Specify the path where the image will be saved
            file.save(img_path)
            # Predict the breed and confidence
            predicted_breed, prediction_confidence = predict_breed(img_path)
            # Render the result page with the predicted breed and confidence
            return render_template('result.html', predicted_breed=predicted_breed, prediction_confidence=prediction_confidence)
    # Render the main page
    return render_template('index.html')

def predict_breed(img_path, confidence_threshold=0.5):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
     # Make predictions
    predictions = model.predict(img_array)
    # Get the predicted class index and confidence
    predicted_class = np.argmax(predictions)
    prediction_confidence = predictions[0][predicted_class]

    if prediction_confidence < confidence_threshold:
        return "Not a dog", int(prediction_confidence * 100)
    # Map the predicted class index to the corresponding breed name
    predicted_breed = dog_breeds[predicted_class]

    return predicted_breed, int(prediction_confidence * 100)
if __name__ == '__main__':
    app.run(debug=True)
