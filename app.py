# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:29:08 2024


@author: mkaab
"""


import os
import sys
sys.path.append(os.path.abspath(r'BLIP'))
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
from flask import Flask, send_from_directory, request, jsonify
from models.blip_itm import blip_itm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from deep_translator import GoogleTranslator




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# text to image
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'




app = Flask(__name__)


@app.route('/')
def index():
    return send_from_directory('static', 'main.html')

@app.route('/page3')
def page3():
    return send_from_directory('static', 'page3.html')


@app.route('/page5')
def page5():
    return send_from_directory('static', 'page5.html')




@app.route('/similarity', methods=['POST'])
def similarity_btw_text():
    data = request.json
    sentences = data.get('sentences', [])
    
    print("Received sentences:", sentences)
    
    
    translated_sentences = []
    for sentence in sentences:
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(sentence) 
            translated_sentences.append(translated_text)
                      
        except Exception as e:
            print(f"Error translating sentence '{sentence}': {e}")
            translated_sentences.append(sentence) 
            

    print("Translated sentences:", translated_sentences)
    
    sentiments = []

    for translated_text in translated_sentences:
        blob = TextBlob(translated_text)
        text_translated = blob.sentiment.polarity
        print(text_translated)
        
        if text_translated>0:
            emotion = 'positive'  
            
        elif text_translated<0:
            emotion = 'negative'            
        else:
            emotion = 'valence'
            
        sentiments.append(emotion)
        
    print("Emotion of sentences:", sentiments)
    
    

    num_sentences = len(translated_sentences)    
    model_sentence = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_sentence.encode(translated_sentences)
    matrix = util.pytorch_cos_sim(embeddings, embeddings)

    matrix = matrix.cpu().numpy()
    
    matrix = 1-matrix
    matrix = np.clip(matrix, a_min=0, a_max=None)

    print(matrix)
    

    
    # Calculate serial scores
    serial_score = []
    for i in range(num_sentences):
        total_score = sum(matrix[i][j] for j in range(i))
        forward_flow_score = total_score / i if i > 0 else total_score
        serial_score.append(forward_flow_score)

    print("Serial scores:", serial_score)
    
    return jsonify({'matrix': matrix.tolist(), 'serial_score': serial_score, 'sentiments': sentiments})





def load_demo_image(image_size,device, img_url):
    raw_image = Image.open(img_url).convert('RGB')


    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

@app.route('/image_text', methods=['POST'])
def img_text():
    
    data = request.json
    image_url = data['image']
    image_url = "static/" + image_url 



    
    sentences = data['sentences']
    print(sentences)
    
    image_size = 384
    image = load_demo_image(image_size=image_size,device=device, img_url = image_url)

    
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    #model = model.to(device='cpu')
    #model = model.to('cuda')

    #caption = 'a cute kitten with orange color'
    #print('text: %s' %sentences)
    score = []
    for sentence in sentences:
        itm_output = model(image,sentence,match_head='itm')
        #itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()  # Convert tensor to Python float
        print('The image and text is matched with a probability of %.4f'%itm_score)
        score.append(itm_score)
        
    return jsonify(scores=score)



if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000)



