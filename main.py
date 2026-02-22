# main.py
from src.sonification import sonify
from src.preprocessing import load_and_preprocess, save_preprocessed
import os


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    input_file = "data/portfolio_data.csv"
    preprocessed_file = "data/preprocessed_amzn.csv"
    output_wav = "output/amzn_sonification.wav"

    df = load_and_preprocess(input_file, column='AMZN', event_limit=0.10)
    save_preprocessed(df, preprocessed_file)

    # Генерируем WAV
    sonify(preprocessed_file, output_wav)