import numpy as np
from flask import Flask, request, jsonify
import io
from PIL import Image
import pytesseract
import cv2
from NLP import model
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if request.files:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}, 400)
        else:
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))
            pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
            txt = open(r'drugs.txt', 'r')
            drugs = [name.split()[0] for name in txt.readlines()]

            def lcs_length(s1, s2):
                m = len(s1)
                n = len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]

                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i - 1] == s2[j - 1]:
                            dp[i][j] = dp[i - 1][j - 1] + 1
                        else:
                            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

                return dp[m][n]

            def LCS(given_name, drugs):
                given_name = str(given_name).lower()
                closest_name = None
                max_lcs_length = 0

                for name in drugs:
                    length = lcs_length(given_name.lower(), str(name).lower())  # Calculate LCS length
                    if length > max_lcs_length:
                        max_lcs_length = length
                        closest_name = name

                return closest_name
            img = np.array(image)
            img = cv2.resize(img, (1000, 400))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            text = pytesseract.image_to_string(img).lower()
            response = LCS(text, drugs)

            return jsonify({'Medicine Name': response})

    elif request.form:
        message = request.form["message"]
        response = model.responde(message)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No file or text provided'}, 400)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
