# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:29:08 2024


@author: mkaab
"""



from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from transformers import BlipProcessor, BlipModel
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
import base64
import io
from flask import Flask, send_from_directory, request, jsonify
import os
import sys
from multiprocessing import Pool
from functools import partial
sys.path.append(os.path.abspath(r'C:/Users/mkaab/BLIP'))
#from blip_itm import blip_itm
#from googletrans import Translator  # googletrans==4.0.0-rc1    this version works 
import requests
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from deep_translator import GoogleTranslator

from flask import Flask, send_from_directory, request, jsonify

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# text to image
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
from models.blip_itm import blip_itm
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'


# text to text
import BERTSimilarity.BERTSimilarity as bertsimilarity
bertsimilarity=bertsimilarity.BERTSimilarity()




def load_demo_image(image_size,device, img_url):
    #img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    #img_url = r"D:\fflow 2\static\img2.jpg"
    raw_image = Image.open(img_url).convert('RGB')

    w,h = raw_image.size
    display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image



app = Flask(__name__)


@app.route('/')
def index():
    return send_from_directory('static', 'main.html')

# Route to serve page3.html
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
    
    #translator = Translator()
    
    translated_sentences = []
    for sentence in sentences:
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(sentence) 
            #translated_text  = translated.text
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
    #better but takes time approx 2 mins
    """
    matrix = [[0] * num_sentences for _ in range(num_sentences)]
    
    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    
    for i in range(num_sentences):
        for j in range(i + 1, num_sentences):
            try:
                score = bertsimilarity.calculate_distance(translated_sentences[i], translated_sentences[j])
                matrix[i][j] = score
                matrix[j][i] = score  # Fill the symmetric element
            except Exception as e:
                print(f"Error calculating similarity between '{translated_sentences[i]}' and '{translated_sentences[j]}': {e}")
    """
    
    
    #faster within 5 seconds
    model_sentence = SentenceTransformer("all-MiniLM-L6-v2")
    #translated_sentences = ['toaster', 'hot' , 'bread', 'plug', 'metal', 'knife', 'spoon', 'jelly', 'jam', 'butter']
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


#@app.route('/image_text', methods=['POST'])
def image_text_similarity():
    #print("working")
    data = request.json
    image_url = data['image']
    image_url = "static/" + image_url 
    #image_url = r"static\img4.jpg"
    img = Image.open(image_url)

# Display image
    #img.show()
    
    sentences = data['sentences']
    print(image_url)
    print(sentences)

    if not os.path.isfile(image_url):
        raise FileNotFoundError(f"Image file {image_url} not found")
    
    # Process the image
    image = Image.open(image_url)
    #print("working2")

    # Print image information to the console
    print(f"Image URL: {image_url}")
    print(f"Image format: {image.format}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")

    # Assuming `processor` and `model` are defined and properly initialized
    image_inputs = processor(images=image, return_tensors="pt")

    # Calculate similarity scores for each sentence
    scores = []
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)
        for sentence in sentences:
            sentence_inputs = processor(text=sentence, return_tensors="pt")
            sentence_embeddings = model.get_text_features(**sentence_inputs)
            similarity_score = cosine_similarity(image_embeddings, sentence_embeddings).item()
            scores.append(similarity_score)

    return jsonify(scores=scores)


@app.route('/image_text', methods=['POST'])
def img_text():
    
    data = request.json
    image_url = data['image']
    image_url = "static/" + image_url 
    #image_url = r"static\img4.jpg"
    #img = Image.open(image_url)

# Display image
    #img.show()
    
    sentences = data['sentences']
    #print(image_url)
    print(sentences)
    
    image_size = 384
    image = load_demo_image(image_size=image_size,device=device, img_url = image_url)

    
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    #model = model.to(device='cpu')
    model = model.to('cuda')

    #caption = 'a cute kitten with orange color'
    #print('text: %s' %sentences)
    score = []
    for sentence in sentences:
        itm_output = model(image,sentence,match_head='itm')
        #itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].item()  # Convert tensor to Python float
        print('The image and text is matched with a probability of %.4f'%itm_score)
        score.append(itm_score)
        

    
    
    
    #itc_score = model(image,sentences,match_head='itc')
    #print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)


    return jsonify(scores=score)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



