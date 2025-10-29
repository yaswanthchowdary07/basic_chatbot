import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

df = pd.read_csv("chat_eng.csv")   # your dataset file

X = df['text']
y = df['intent']

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

clf = MultinomialNB()
clf.fit(X_vec, y)

responses = {
    "greeting": ["Hello! Ready to study English?", "Hi there! Let’s explore English lessons together!"],
    "bye": ["Goodbye! Keep revising English daily!", "See you later, and don’t forget to practice reading!"],
    "tips": ["Read the passages carefully, practice grammar, and revise poems regularly."],
    "syllabus": ["Here are the TS Board Class 10 English Units:\n"
                 "1. Personality Development\n2. Wit and Humor\n3. Human Relations\n4. Films and Theatre\n"
                 "5. Social Issues\n6. Bio-Diversity\n7. Nation and Diversity\n8. Human Rights"],

    "chapter_1": ["Unit 1 is *Personality Development*."],
    "topics_ch1": ["Lessons:\n1. The Selfless Service\n2. Poem: Be the Best"],
    "resources_ch1": ["Resources: TS Board Textbook, NCERT extracts, YouTube: English Explained"],

    "chapter_2": ["Unit 2 is *Wit and Humor*."],
    "topics_ch2": ["Lessons:\n1. The Open Window\n2. Poem: The Laughing Song"],
    "resources_ch2": ["Resources: TS English materials, Vedantu English, NCERT Solutions"],

    "chapter_3": ["Unit 3 is *Human Relations*."],
    "topics_ch3": ["Lessons:\n1. Two Friends\n2. Poem: Friendship"],
    "resources_ch3": ["Resources: Class notes, RS Aggarwal English Guide"],

    "chapter_4": ["Unit 4 is *Films and Theatre*."],
    "topics_ch4": ["Lessons:\n1. An Inspector Calls (extract)\n2. Poem: The Curtain Rises"],
    "resources_ch4": ["Resources: YouTube Plays, English Teacher Notes"],

    "chapter_5": ["Unit 5 is *Social Issues*."],
    "topics_ch5": ["Lessons:\n1. The Storeyed House (Part 1)\n2. The Storeyed House (Part 2)\n3. Poem: Abandoned"],
    "resources_ch5": ["Resources: Byju’s English, NCERT Supplementary, TS Study Material"],

    "chapter_6": ["Unit 6 is *Bio-Diversity*."],
    "topics_ch6": ["Lessons:\n1. The Banyan\n2. Poem: Songs of Nature"],
    "resources_ch6": ["Resources: NCERT Biodiversity passages, Class notes"],

    "chapter_7": ["Unit 7 is *Nation and Diversity*."],
    "topics_ch7": ["Lessons:\n1. India: A Nation of Many\n2. Poem: Unity in Diversity"],
    "resources_ch7": ["Resources: TS Board textbook, Khan Academy English, Teacher notes"],

    "chapter_8": ["Unit 8 is *Human Rights*."],
    "topics_ch8": ["Lessons:\n1. Rights and Responsibilities\n2. Poem: Freedom"],
    "resources_ch8": ["Resources: Human Rights articles, English notes, TS Textbook"],

    "resources_general": ["For English, use: TS Board Textbook, NCERT Supplementary Reader, Vedantu English, and teacher’s notes."]
}

intent_keywords = {
    "greeting": ["hi", "hello", "hey"],
    "bye": ["bye", "goodbye", "see you"],
    "tips": ["tips", "advice", "study help"],
    "syllabus": ["syllabus", "units", "chapters", "english"],
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

    for i in range(1, 9):
        if f"chapter {i}" in user_input_lower or f"unit {i}" in user_input_lower:
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
