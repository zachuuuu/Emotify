import os
import torch
import pandas as pd
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

CSV_PATH = '../datasets/DEAM/DEAM_Annotations/static_annotations_averaged_songs.csv'
AUDIO_DIR = '../datasets/DEAM/DEAM_audio'
OUTPUT_FILE = 'deam_mert_features.parquet'

TARGET_SR = 24000
MAX_DURATION_SEC = 30

def get_device():
    """
        Choosing device to use
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()

def load_and_preprocess_audio(file_path, target_sr=24000, max_seconds=30):
    """
        Load data with helping of librosa
    """
    try:
        audio_array, _ = librosa.load(file_path, sr=target_sr, mono=True)

        max_length = target_sr * max_seconds
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]

        if len(audio_array) < (target_sr * 0.5):
            return None

        return torch.tensor(audio_array, dtype=torch.float32)
    except Exception as e:
        print(f"Error with reading file: {e}")
        return None


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.head(40).copy()

    try:
        df['song_id'] = df['song_id'].fillna(0).astype(int).astype(str)
    except:
        df['song_id'] = df['song_id'].astype(str)

    print(f'Songs in file: {len(df)}')

    model_id = "m-a-p/MERT-v1-330M"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    model.to(device)
    model.eval()

    embeddings_list = []
    valid_indices = []
    missing_cnt = 0

    print(f"Working on device: {device}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        song_id = row['song_id']
        file_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")

        if not os.path.exists(file_path):
            missing_cnt += 1
            continue

        audio_input = load_and_preprocess_audio(file_path, target_sr=TARGET_SR, max_seconds=MAX_DURATION_SEC)

        if audio_input is None:
            continue

        inputs = feature_extractor(audio_input, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        inputs = inputs.to(device)

        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.last_hidden_state

                pooled_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()

                embeddings_list.append(pooled_embedding)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error with song {song_id}: {e}")
                continue

    final_df = df.loc[valid_indices].reset_index(drop=True)
    final_df['mert_features'] = list(embeddings_list)

    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"DataFrame saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()