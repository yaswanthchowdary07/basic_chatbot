import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random


df = pd.read_csv("chat_bio.csv")

X = df['text']
y = df['intent']


vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

clf = LogisticRegression(max_iter=500)
clf.fit(X_vec, y)


responses = {
    "greeting": ["Hello! How can I help you with Biology today?", "Hi there! Ready to explore Biology?"],
    "bye": ["Goodbye! Keep revising Biology concepts!", "See you later, best of luck with your Biology!"],
    "tips": ["Revise diagrams daily, practice definitions, and focus on SCERT/NCERT-style questions!"],
    "syllabus": ["Here are the Class 10 Biology chapters (Telangana Board):\n"
                 "1. Nutrition\n2. Respiration\n3. Transportation\n4. Excretion\n5. Control & Coordination\n"
                 "6. Reproduction\n7. Coordination in Life Processes\n8. Heredity & Evolution\n9. Our Environment"],

    "chapter_1": ["Chapter 1 is *Nutrition in Plants & Animals*."],
    "topics_ch1": ["Topics:\n1. Autotrophic & heterotrophic nutrition\n2. Photosynthesis\n3. Human digestive system"],
    "resources_ch1": ["Resources: YouTube: TS Board Biology, Khan Academy\nNotes: NCERT Solutions, Telangana SCERT PDFs"],

    "chapter_2": ["Chapter 2 is *Respiration*."],
    "topics_ch2": ["Topics:\n1. Types of respiration\n2. Human respiratory system\n3. Mechanism of breathing\n4. Aerobic & anaerobic respiration"],
    "resources_ch2": ["Resources: YouTube: Biology Wallah, TS SCERT\nNotes: Lakhmir Singh Biology, NCERT Exemplar"],

    "chapter_3": ["Chapter 3 is *Transportation*."],
    "topics_ch3": ["Topics:\n1. Circulatory system\n2. Structure of heart\n3. Blood & lymph circulation\n4. Transport in plants"],
    "resources_ch3": ["Resources: YouTube: Khan Academy, TS Board Lectures\nNotes: SCERT Notes, NCERT Solutions"],

    "chapter_4": ["Chapter 4 is *Excretion*."],
    "topics_ch4": ["Topics:\n1. Excretory organs in humans\n2. Structure & function of nephron\n3. Excretion in plants"],
    "resources_ch4": ["Resources: YouTube: TS Board Biology, ExamFear Education\nNotes: Telangana SCERT PDFs, NCERT Solutions"],

    "chapter_5": ["Chapter 5 is *Control & Coordination*."],
    "topics_ch5": ["Topics:\n1. Human nervous system\n2. Reflex action\n3. Hormonal coordination in humans & plants"],
    "resources_ch5": ["Resources: YouTube: Biology Wallah, Khan Academy\nNotes: NCERT Solutions, Lakhmir Singh Biology"],

    "chapter_6": ["Chapter 6 is *Reproduction*."],
    "topics_ch6": ["Topics:\n1. Asexual reproduction\n2. Sexual reproduction in plants\n3. Reproduction in humans"],
    "resources_ch6": ["Resources: YouTube: TS Board Biology, Vedantu 9&10\nNotes: SCERT Notes, NCERT PDFs"],

    "chapter_7": ["Chapter 7 is *Coordination in Life Processes*."],
    "topics_ch7": ["Topics:\n1. Coordination in digestive, respiratory & circulatory systems\n2. Homeostasis"],
    "resources_ch7": ["Resources: YouTube: TS SCERT, ExamFear\nNotes: NCERT Exemplar, State Board PDFs"],

    "chapter_8": ["Chapter 8 is *Heredity & Evolution*."],
    "topics_ch8": ["Topics:\n1. Mendel’s experiments\n2. Laws of inheritance\n3. Evolutionary concepts"],
    "resources_ch8": ["Resources: YouTube: Khan Academy, Biology Wallah\nNotes: Telangana SCERT, NCERT Solutions"],

    "chapter_9": ["Chapter 9 is *Our Environment*."],
    "topics_ch9": ["Topics:\n1. Ecosystem & food chains\n2. Ozone depletion\n3. Waste management\n4. Human impact on environment"],
    "resources_ch9": ["Resources: YouTube: TS Board Biology, Vedantu\nNotes: NCERT Solutions, Byju’s PDFs"],

    "resources_general": ["Use Telangana SCERT, NCERT Exemplar, Biology Wallah, and Khan Academy for Class 10 Biology."]
}


intent_keywords = {
    "greeting": ["hi", "hello", "hey"],
    "bye": ["bye", "goodbye", "see you"],
    "tips": ["tips", "advice", "study help"],
    "syllabus": ["syllabus", "chapters", "biology class 10"],
    "resources_general": ["resources", "materials", "notes", "references"]
}

def chatbot_response(user_input):
    user_input_lower = user_input.lower()

    
    X_test = vectorizer.transform([user_input])
    proba = clf.predict_proba(X_test)[0]
    max_prob = max(proba)
    intent = clf.predict(X_test)[0]

    print(f"Predicted: {intent} Confidence: {max_prob}")

    if max_prob >= 0.1:  
        if intent in responses:
            return f"Chatbot (ml): {random.choice(responses[intent])}"

    
    for intent, keywords in intent_keywords.items():
        if any(word in user_input_lower for word in keywords):
            return f"Chatbot (rule): {random.choice(responses[intent])}"

    for i in range(1, 10):
        if f"chapter {i}" in user_input_lower:
            if "topics" in user_input_lower:
                return responses.get(f"topics_ch{i}", ["Sorry, topics not available."])[0]
            elif "resources" in user_input_lower:
                return responses.get(f"resources_ch{i}", ["Sorry, resources not available."])[0]
            else:
                return responses.get(f"chapter_{i}", ["Sorry, chapter info not available."])[0]

    return "Sorry, I didn’t understand. Can you rephrase?"


while True:
    user_inp = input("You: ")
    if user_inp.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    print(chatbot_response(user_inp))
