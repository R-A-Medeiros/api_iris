from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Carregando modelo KNN
knn_model = joblib.load('C:\Users\1221316963\PycharmProjects\api_ris\modelo_knn(92%).pkl')

scaler = StandardScaler()

# Dicionario para mapear o nome das especies de acordo com a saída numérica
species_mapping = {
    0: 'Iris Setosa',
    1: 'Iris Versicolor',
    2: 'Iris Virginica'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Iris')
def iris():
    return render_template('about-iris.html')


@app.route('/About')
def about():
    return render_template('about-us.html')

@app.route('/predict', methods=['POST', ])
def predict():
    if request.method == 'POST':
        sepalLength = float(request.form['sepalLength'])
        sepalWidth = float(request.form['sepalWidth'])
        petalLenght = float(request.form['petalLength'])
        petalWidth = float(request.form['petalWidth'])

        # Normalizando os dados de entrada
        input_data = scaler.transform([[sepalLength, sepalWidth, petalLenght, petalWidth]])

        # Previsão
        prediction = knn_model.predict(input_data)

        # Mapeando a saída numérica para o nome da espécie correspondente
        speceis_result = species_mapping.get(prediction[0], 'Espécie Desconhecida')

        return render_template('index.html', result=speceis_result)


if __name__ == '__main__':
    app.run(debug=True)
