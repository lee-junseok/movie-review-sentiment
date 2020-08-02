# Movie-Review-Sentiment

Sentiment analysis model on movie reviews trained on the IMDB dataset.

- Used *Keras Sequential model* with two fully-connected layers.

- Used *SpaCy 'en_core_web_sm' model* and tokenized using *tf.keras.preprocessing.text.Tokenizer*.

Training work in `imdb-movie-sentiment.ipynb`.

Trained tokenizer and model are saved in `tokenizer.pickle` and `my_model.h5`.

To test the model, run:
```
conda create -n sentiment python=3.7.7 pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
```
python movie-review-sentiment.py
```
and type a review sentence.
