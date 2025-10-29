import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random


df = pd.read_csv("chat_phy.csv")

X = df['text']
y = df['intent']


vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)


clf = LogisticRegression(max_iter=500)
clf.fit(X_vec, y)


responses = {
    "greeting": ["Hello! How can I help you with physics today?", "Hi there! Ready for some learning?"],
    "bye": ["Goodbye! Keep studying hard!", "See you later, best of luck with your physics!"],
    "tips": ["Study a little every day, revise regularly, and practice past papers!"],
    "syllabus": ["Here are the Class 10 Physics chapters:\n"
                 "1. Light\n2. Human Eye & Colourful World\n3. Electricity\n4. Magnetic Effects of Current\n"
                 "5. Sources of Energy\n6. Gravitation & Motion\n7. Work, Energy & Power\n8. Sound\n9. Structure of Atom"],

    "chapter_1": ["Chapter 1 is *Light – Reflection & Refraction*."],
    "topics_ch1": ["Topics:\n1. Reflection of light at plane and curved surfaces\n2. Laws of reflection\n3. Image formation by spherical mirrors\n4. Refraction of light through lenses\n5. Lens formula & magnification"],
    "resources_ch1": ["Resources: YouTube: Vedantu 9&10, Physics Wallah\nNotes: NCERT Solutions, TopperLearning PDFs"],

    "chapter_2": ["Chapter 2 is *Human Eye & Colourful World*."],
    "topics_ch2": ["Topics:\n1. Structure & working of human eye\n2. Defects of vision & corrections\n3. Refraction through prism\n4. Dispersion & rainbow formation"],
    "resources_ch2": ["Resources: YouTube: Khan Academy, Physics Wallah\nNotes: NCERT Exemplar, Lakhmir Singh Physics"],

    "chapter_3": ["Chapter 3 is *Electricity*."],
    "topics_ch3": ["Topics:\n1. Electric current, potential difference, resistance\n2. Ohm’s law & verification\n3. Series & parallel circuits\n4. Heating effect of current\n5. Electrical energy & power"],
    "resources_ch3": ["Resources: YouTube: Physics Wallah, ExamFear Education\nNotes: NCERT Solutions, Arihant Guide"],

    "chapter_4": ["Chapter 4 is *Magnetic Effects of Current*."],
    "topics_ch4": ["Topics:\n1. Magnetic field due to conductor\n2. Ampere’s law applications\n3. Electromagnetic induction\n4. Electric motor & generator"],
    "resources_ch4": ["Resources: YouTube: Khan Academy, Physics Wallah\nNotes: NCERT Solutions, TopperLearning"],

    "chapter_5": ["Chapter 5 is *Sources of Energy*."],
    "topics_ch5": ["Topics:\n1. Conventional sources: coal, petroleum, gas\n2. Non-conventional sources: solar, wind, hydro, nuclear\n3. Pros & cons of energy sources"],
    "resources_ch5": ["Resources: YouTube: Vedantu 9&10, ExamFear Education\nNotes: NCERT Book, Byju’s PDFs"],

    "chapter_6": ["Chapter 6 is *Gravitation & Motion*."],
    "topics_ch6": ["Topics:\n1. Universal law of gravitation\n2. Acceleration due to gravity\n3. Motion in 1D & 2D\n4. Newton’s laws of motion"],
    "resources_ch6": ["Resources: YouTube: Physics Wallah, ExamFear Education\nNotes: NCERT Solutions, Lakhmir Singh Guide"],

    "chapter_7": ["Chapter 7 is *Work, Energy & Power*."],
    "topics_ch7": ["Topics:\n1. Work done by force\n2. Kinetic & potential energy\n3. Work-energy theorem\n4. Power & efficiency"],
    "resources_ch7": ["Resources: YouTube: Vedantu 9&10, Physics Wallah\nNotes: NCERT Exemplar, TopperLearning"],

    "chapter_8": ["Chapter 8 is *Sound*."],
    "topics_ch8": ["Topics:\n1. Nature & propagation of sound\n2. Characteristics of sound waves\n3. Reflection of sound – echo\n4. Musical instruments"],
    "resources_ch8": ["Resources: YouTube: Khan Academy, Physics Wallah\nNotes: NCERT Book, Byju’s PDFs"],

    "chapter_9": ["Chapter 9 is *Structure of Atom*."],
    "topics_ch9": ["Topics:\n1. Subatomic particles\n2. Thomson & Rutherford models\n3. Bohr’s model of atom\n4. Atomic number, isotopes, isobars"],
    "resources_ch9": ["Resources: YouTube: Vedantu 9&10, Physics Wallah\nNotes: NCERT Solutions, Arihant PDFs"],

    "resources_general": ["You can use NCERT Solutions, Physics Wallah, Vedantu 9&10, and TopperLearning PDFs for Class 10 Physics."]
}


intent_keywords = {
    "greeting": ["hi", "hello", "hey"],
    "bye": ["bye", "goodbye", "see you"],
    "tips": ["tips", "advice", "study help"],
    "syllabus": ["syllabus", "chapters", "class 10 physics"],
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
