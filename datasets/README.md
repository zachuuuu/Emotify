# DEAM dataset
- [Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music/)
- annotation_transformer.py is created to connect to csv files by valence_mean, valence_std, arousal_mean, arousal_std columns

# MTG
- replace ".mp3" to ".npy": `sed -i '' 's/\.mp3/.npy/g' datasets/MTG/autotagging_moodtheme.tsv`