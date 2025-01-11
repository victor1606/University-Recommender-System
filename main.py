from flask import Flask, render_template, request, redirect, url_for
from serpapi import GoogleSearch
import pandas as pd

app = Flask(__name__)

def calculate_scores(user_data):
    # Automatically map subjects to their respective fields
    subjects = {
        user_data.get('Subiect ea', ''): user_data.get('NOTA_FINALA_EA', 0),
        user_data.get('Subiect ec', ''): user_data.get('NOTA_FINALA_EC', 0),
        user_data.get('Subiect ed', ''): user_data.get('NOTA_FINALA_ED', 0)
    }

    # Extract other relevant fields
    digital_skills = user_data.get('PUNCTAJ DIGITALE', 0)
    communication_skills = user_data.get('ORAL_PMO', 0)
    overall_grade = user_data.get('Medie', 0)
    profile = user_data.get('Profil', '')
    specialization = user_data.get('Specializare', '')

    # Initialize subject-based variables
    math_grade = 0
    science_grade = 0
    language_grade = 0
    history_grade = 0
    art_grade = 0
    sports_grade = 0
    vocational_grade = 0

    # Assign grades based on subject mapping
    for subject, grade in subjects.items():
        if 'Matematică' in subject:
            math_grade = grade
        elif any(sci in subject for sci in ['Biologie', 'Chimie', 'Fizică', 'Anatomie']):
            science_grade = grade
        elif 'Limba română' in subject:
            language_grade = grade
        elif any(hist in subject for hist in ['Istorie', 'Geografie']):
            history_grade = grade
        elif any(art in subject for art in ['Arte', 'Muzică', 'Desen']):
            art_grade = grade
        elif any(sport in subject for sport in ['Educație fizică', 'Sport']):
            sports_grade = grade
        elif any(voc in subject for voc in ['Tehnic', 'Servicii', 'Resurse naturale']):
            vocational_grade = grade

    # Penalty for low overall grade
    difficulty_penalty = 1 if overall_grade >= 8 else 0.8 if overall_grade >= 6 else 0.5

    # Profile-based penalty or boost
    profile_adjustments = {
        'Uman': {'Computer Science': 0.2, 'Technical': 0.3, 'Medicine': 0.4, 'Psychology': 1.2, 'Political Science': 1.2, 'Journalism': 1.3, 'Arts': 1.4},
        'Real': {'Law': 0.4, 'Psychology': 0.5, 'Arts': 0.3},
        'Tehnologică': {'Medicine': 0.5, 'Law': 0.6, 'Arts': 0.4},
        'Vocatională': {'Engineering': 0.5, 'Computer Science': 0.5}
    }

    profile_penalty = profile_adjustments.get(profile, {})

    # Scores for each field with penalty applied
    scores = {}

    # Architecture
    scores['Architecture'] = difficulty_penalty * (0.4 * math_grade + 0.3 * art_grade + 0.2 * digital_skills + 0.1 * overall_grade) * profile_penalty.get('Architecture', 1)
    
    # Law
    scores['Law'] = difficulty_penalty * (0.5 * language_grade + 0.3 * history_grade + 0.1 * overall_grade + 0.1 * communication_skills) * profile_penalty.get('Law', 1)

    # Geography
    scores['Geography'] = difficulty_penalty * (0.5 * history_grade + 0.3 * communication_skills + 0.2 * overall_grade)
    
    # Journalism
    scores['Journalism'] = difficulty_penalty * (0.5 * language_grade + 0.4 * communication_skills + 0.1 * overall_grade) * profile_penalty.get('Journalism', 1)

    # Psychology
    scores['Psychology'] = difficulty_penalty * (0.4 * science_grade + 0.3 * language_grade + 0.2 * overall_grade + 0.1 * communication_skills) * profile_penalty.get('Psychology', 1)

    # Political Science (SNSPA)
    scores['Political Science'] = difficulty_penalty * (0.5 * language_grade + 0.3 * history_grade + 0.2 * overall_grade) * profile_penalty.get('Political Science', 1)
    
    # Medicine
    scores['Medicine'] = difficulty_penalty * (0.6 * science_grade + 0.2 * overall_grade + 0.2 * digital_skills) * profile_penalty.get('Medicine', 1)

    # Computer Science
    scores['Computer Science'] = difficulty_penalty * (0.6 * math_grade + 0.3 * digital_skills + 0.1 * overall_grade) * profile_penalty.get('Computer Science', 1)

    # Mathematics
    scores['Mathematics'] = difficulty_penalty * (0.7 * math_grade + 0.2 * overall_grade + 0.1 * science_grade)

    # Physical Education & Sports
    scores['Physical Education'] = difficulty_penalty * (0.6 * sports_grade + 0.2 * overall_grade + 0.2 * communication_skills)

    # Technical Fields
    scores['Technical'] = difficulty_penalty * (0.6 * vocational_grade + 0.3 * math_grade + 0.1 * digital_skills) * profile_penalty.get('Technical', 1)

    # Arts
    scores['Arts'] = difficulty_penalty * (0.6 * art_grade + 0.3 * language_grade + 0.1 * overall_grade) * profile_penalty.get('Arts', 1)

    # Recommend the best field
    recommended_field = max(scores, key=scores.get)
    
    return recommended_field, scores

def search_universities_near_city(city, interest):
    params = {
        "engine": "google",
        "q": f"universities near {city} Romania {interest}",
        "location": f"{city}, Romania",
        "num": 10,
        "api_key": "9e0409085db46c894a9646b2d03041e2954fc586212772d70c2ab195e435dd69"
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    keywords = ["university", "universitate", "faculty", "facultate", "institute", "institut", "college", "colegiu"]
    universities = []
    for result in results.get('organic_results', []):
        title = result.get('title', '').lower()
        link = result.get('link')
        if any(keyword in title for keyword in keywords):
            universities.append({'Title': result.get('title'), 'Link': link})
    return universities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_data = {
            'Subiect ea': request.form['subiect_ea'],
            'Subiect ec': request.form['subiect_ec'],
            'Subiect ed': request.form['subiect_ed'],
            'NOTA_FINALA_EA': float(request.form['nota_finala_ea']),
            'NOTA_FINALA_EC': float(request.form['nota_finala_ec']),
            'NOTA_FINALA_ED': float(request.form['nota_finala_ed']),
            'PUNCTAJ DIGITALE': float(request.form['punctaj_digitale']),
            'ORAL_PMO': float(request.form['oral_pmo']),
            'Medie': float(request.form['medie']),
            'Profil': request.form['profil'],
            'Specializare': request.form['specializare']
        }
        recommendation, scores = calculate_scores(user_data)
        universities = search_universities_near_city(request.form['city'], recommendation)
        return render_template('result.html', recommendation=recommendation, scores=scores, universities=universities)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)