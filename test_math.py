import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

df = pd.read_csv("chat_math.csv")

X = df['text']
y = df['intent']

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

clf = MultinomialNB()
clf.fit(X_vec, y)

responses = {
    "greeting": ["Hello! Ready to solve some math problems?", "Hi there! Let's explore mathematics together!"],
    "bye": ["Goodbye! Keep practicing math daily!", "See you later, and don’t forget to revise formulas!"],
    "tips": ["Practice problems regularly, revise formulas daily, and solve previous year papers!"],
    "syllabus": ["Here are the Class 10 Mathematics chapters:\n"
                 "1. Real Numbers\n2. Polynomials\n3. Pair of Linear Equations\n4. Quadratic Equations\n"
                 "5. Arithmetic Progression\n6. Triangles\n7. Coordinate Geometry\n8. Trigonometry\n"
                 "9. Circles\n10. Statistics & Probability"],

    "chapter_1": ["Chapter 1 is *Real Numbers*."],
    "topics_ch1": ["Topics:\n1. Euclid’s Division Lemma\n2. Fundamental Theorem of Arithmetic\n3. HCF & LCM"],
    "resources_ch1": ["Resources: YouTube: Vedantu Math, Khan Academy\nNotes: NCERT Solutions, RS Aggarwal"],

    "chapter_2": ["Chapter 2 is *Polynomials*."],
    "topics_ch2": ["Topics:\n1. Zeros of polynomials\n2. Relationship between coefficients and roots\n3. Division algorithm"],
    "resources_ch2": ["Resources: YouTube: MathonGo, ExamFear Education\nNotes: NCERT Exemplar, RD Sharma"],

    "chapter_3": ["Chapter 3 is *Pair of Linear Equations in Two Variables*."],
    "topics_ch3": ["Topics:\n1. Graphical method\n2. Substitution & elimination methods\n3. Applications"],
    "resources_ch3": ["Resources: YouTube: Khan Academy, Physics Wallah Math\nNotes: NCERT Solutions, RS Aggarwal"],

    "chapter_4": ["Chapter 4 is *Quadratic Equations*."],
    "topics_ch4": ["Topics:\n1. Factorization method\n2. Completing the square\n3. Quadratic formula\n4. Applications"],
    "resources_ch4": ["Resources: YouTube: MathonGo, Vedantu Math\nNotes: NCERT Exemplar, Arihant Guide"],

    "chapter_5": ["Chapter 5 is *Arithmetic Progression*."],
    "topics_ch5": ["Topics:\n1. nth term of AP\n2. Sum of first n terms\n3. Applications in daily life"],
    "resources_ch5": ["Resources: YouTube: Unacademy, ExamFear Education\nNotes: NCERT Solutions, RS Aggarwal"],

    "chapter_6": ["Chapter 6 is *Triangles*."],
    "topics_ch6": ["Topics:\n1. Similar triangles\n2. Pythagoras theorem\n3. Applications"],
    "resources_ch6": ["Resources: YouTube: Vedantu Math, Khan Academy\nNotes: NCERT Exemplar, RD Sharma"],

    "chapter_7": ["Chapter 7 is *Coordinate Geometry*."],
    "topics_ch7": ["Topics:\n1. Distance formula\n2. Section formula\n3. Area of triangle"],
    "resources_ch7": ["Resources: YouTube: ExamFear Education, MathonGo\nNotes: NCERT Solutions, RS Aggarwal"],

    "chapter_8": ["Chapter 8 is *Trigonometry*."],
    "topics_ch8": ["Topics:\n1. Trigonometric ratios\n2. Trigonometric identities\n3. Heights and distances"],
    "resources_ch8": ["Resources: YouTube: Vedantu Math, Khan Academy\nNotes: NCERT Exemplar, RD Sharma"],

    "chapter_9": ["Chapter 9 is *Circles*."],
    "topics_ch9": ["Topics:\n1. Tangent to a circle\n2. Number of tangents from a point"],
    "resources_ch9": ["Resources: YouTube: MathonGo, ExamFear Education\nNotes: NCERT Solutions, RS Aggarwal"],

    "chapter_10": ["Chapter 10 is *Statistics & Probability*."],
    "topics_ch10": ["Topics:\n1. Mean, median, mode\n2. Probability of events\n3. Applications"],
    "resources_ch10": ["Resources: YouTube: Vedantu Math, Khan Academy\nNotes: NCERT Book, RS Aggarwal"],

    "resources_general": ["You can use NCERT Solutions, RS Aggarwal, RD Sharma, Vedantu, and Khan Academy for Class 10 Mathematics."]
}

intent_keywords = {
    "greeting": ["hi", "hello", "hey"],
    "bye": ["bye", "goodbye", "see you"],
    "tips": ["tips", "advice", "study help"],
    "syllabus": ["syllabus", "chapters", "class 10 math"],
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

    for i in range(1, 11):
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
