import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

def predict_sentiment(input_text):
    encoded_text = tokenizer(input_text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    labels = ['Negative', 'Neutral', 'Positive']
    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        st.write(l, s)
    for i in range(len(scores)):
        if scores[i] == max(scores):
            label = labels[i]
            break
    if label == "Positive":
        st.write('Overall Sentiment: ', label, '🙂')
    if label == "Negative":
        st.write('Overall Sentiment: ', label, '🙁')
    if label == "Neutral":
        st.write('Overall Sentiment: ', label, '😐')
        


st.header('Sentiment Analysis 😎', divider='rainbow')

with st.form("my_form"):
    # st.write("Enter Your Text Here ")
    input_text = st.text_input('Enter Your Text Here')
    submitted = st.form_submit_button("Submit", type='primary')
    if submitted:
       predict_sentiment(input_text)



    

