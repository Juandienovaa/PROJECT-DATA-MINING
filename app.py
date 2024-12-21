from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import os

app = Flask(__name__)

# Dataframe untuk menyimpan data
df = pd.DataFrame(columns=['Bulan', 'Penjualan', 'Stok_Awal', 'Kategori', 'Bulan_Numerik'])
model = LinearRegression()

# Konversi bulan ke angka
bulan_to_numerik = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
    'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
    'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

def train_model():
    if not df.empty:
        X = df[['Bulan_Numerik', 'Penjualan']]
        y = df['Stok_Awal']
        model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_data', methods=['POST'])
def add_data():
    global df
    try:
        bulan = request.form.get('bulan').strip().capitalize()
        penjualan = int(request.form.get('penjualan', 0))
        stok_awal = int(request.form.get('stok_awal', 0))
        kategori = request.form.get('kategori')

        if bulan not in bulan_to_numerik:
            return jsonify({"status": "error", "message": "Bulan tidak valid."})

        bulan_numerik = bulan_to_numerik[bulan]
        new_data = pd.DataFrame({
            'Bulan': [bulan],
            'Penjualan': [penjualan],
            'Stok_Awal': [stok_awal],
            'Kategori': [kategori],
            'Bulan_Numerik': [bulan_numerik]
        })
        df = pd.concat([df, new_data], ignore_index=True)

        train_model()
        return jsonify({"status": "success", "message": "Data berhasil ditambahkan!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/show_graph', methods=['POST'])
def show_graph():
    if df.empty:
        return jsonify({"status": "error", "message": "Data tidak cukup untuk menampilkan grafik."})

    try:
        graph_urls = {}
        predictions = {}
        explanations = {}

        for kategori in df['Kategori'].unique():
            subset = df[df['Kategori'] == kategori]

            plt.figure(figsize=(10, 6))
            plt.scatter(subset['Bulan_Numerik'], subset['Stok_Awal'], label=f'{kategori} (Data)')

            if len(subset) > 1:
                X = subset[['Bulan_Numerik', 'Penjualan']]
                y = subset['Stok_Awal']
                model.fit(X, y)
                plt.plot(subset['Bulan_Numerik'], model.predict(X), label=f'{kategori} (Prediksi)')

                next_month = subset['Bulan_Numerik'].max() + 1
                predicted_stock = model.predict([[next_month, subset['Penjualan'].mean()]])[0]
                predictions[kategori] = round(predicted_stock, 2)

                explanations[kategori] = (
                    "Stok prediksi bulan depan positif. Tren penjualan stabil."
                    if predicted_stock > 0 else
                    "Stok prediksi bulan depan negatif. Mungkin ada penurunan permintaan."
                )

            plt.title(f'Prediksi Stok - {kategori}', fontsize=16)
            plt.xlabel('Bulan', fontsize=12)
            plt.ylabel('Stok Awal', fontsize=12)
            plt.legend()

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            graph_filename = f"graph_{kategori}_{timestamp}.png"
            graph_path = os.path.join('static', graph_filename)
            plt.savefig(graph_path)
            plt.close()

            graph_urls[kategori] = f"/static/{graph_filename}"

        return jsonify({
            "status": "success",
            "graphs": graph_urls,
            "predictions": predictions,
            "explanations": explanations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
