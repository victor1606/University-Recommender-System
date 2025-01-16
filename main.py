import os
import csv
import sys
import re
import collections
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import requests

import base64

matplotlib.use('Agg')

from flask import Flask, request, render_template
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from scipy import stats
from collections import defaultdict

from io import BytesIO

app = Flask(__name__)

def fig_to_base64(fig):
    """Convert a Matplotlib figure object to a Base64-encoded PNG string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return base64_img

def calculate_scores(user_data):
    try:
        data_bac = pd.read_csv("./BAC CSV/bac2019.csv")
    except FileNotFoundError:
        print("Could not find bac2019.csv. Make sure it exists in BAC CSV/ folder.")
        data_bac = pd.DataFrame()

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
    logic_grade = 0
    info_grade = 0

    # Assign grades based on subject mapping
    for subject, grade in subjects.items():
        if any(math_subj in subject for math_subj in ['Matematică MATE-INFO', 'Matematică ST-NAT', 'Matematică TEHN']):
            math_grade = grade
        elif any(sci in subject for sci in [
            'Biologie vegetală și animală', 'Anatomie și fiziologie umană, genetică și ecologie umană',
            'Chimie organică TEH Nivel I/II', 'Chimie anorganică TEH Nivel I/II',
            'Chimie organică TEO Nivel I/II', 'Chimie anorganică TEO Nivel I/II',
            'Fizică TEO', 'Fizică TEH'
        ]):
            science_grade = grade
        elif any(lang in subject for lang in ['Limba română (UMAN)', 'Limba română (REAL)']):
            language_grade = grade
        elif any(hist in subject for hist in ['Istorie', 'Geografie']):
            history_grade = grade
        elif 'Logică, argumentare și comunicare' in subject:
            logic_grade = grade
        elif 'I' in subject:
            info_grade = grade

    # Penalty for low overall grade
    difficulty_penalty = 1 if overall_grade >= 8 else 0.8 if overall_grade >= 6 else 0.5

    # Profile-based penalty or boost
    profile_adjustments = {
        'Uman': {
            'Computer Science': 0.2, 'Technical': 0.3, 'Medicine': 0.4,
            'Psychology': 1.2, 'Political Science': 1.2,
            'Journalism': 1.3, 'Arts': 1.4
        },
        'Real': {
            'Law': 0.4, 'Psychology': 0.5, 'Arts': 0.3,
            'Computer Science': 1.2
        },
        'Tehnologică': {
            'Medicine': 0.5, 'Law': 0.6, 'Arts': 0.4,
            'Technical': 1.2
        },
        'Vocatională': {
            'Engineering': 0.5, 'Computer Science': 0.5,
            'Arts': 1.2
        }
    }
    profile_penalty = profile_adjustments.get(profile, {})

    # Scores for each field with penalty applied
    scores = {}

    # Architecture
    scores['Architecture'] = difficulty_penalty * (
        0.4 * math_grade + 0.002 * digital_skills + 0.4 * overall_grade
    ) * profile_penalty.get('Architecture', 1)

    # Law
    if logic_grade > 6:
        scores['Law'] = difficulty_penalty * (
            0.4 * language_grade + 0.2 * history_grade +
            0.1 * overall_grade + 0.1 * communication_skills +
            0.2 * logic_grade
        ) * profile_penalty.get('Law', 1)
    else:
        scores['Law'] = difficulty_penalty * (
            0.4 * language_grade + 0.4 * history_grade +
            0.1 * overall_grade + 0.1 * communication_skills
        ) * profile_penalty.get('Law', 1)

    # Geography
    scores['Geography'] = difficulty_penalty * (
        0.5 * history_grade + 0.3 * communication_skills + 0.2 * overall_grade
    )

    # Journalism
    scores['Journalism'] = difficulty_penalty * (
        0.5 * language_grade + 0.4 * communication_skills + 0.1 * overall_grade
    ) * profile_penalty.get('Journalism', 1)

    # Psychology
    scores['Psychology'] = difficulty_penalty * (
        0.3 * science_grade + 0.3 * language_grade +
        0.2 * overall_grade + 0.2 * communication_skills
    ) * profile_penalty.get('Psychology', 1)

    # Political Science (SNSPA)
    scores['Political Science'] = difficulty_penalty * (
        0.5 * language_grade + 0.3 * history_grade + 0.2 * overall_grade
    ) * profile_penalty.get('Political Science', 1)

    # Medicine
    scores['Medicine'] = difficulty_penalty * (
        0.6 * science_grade + 0.3 * overall_grade + 0.001 * digital_skills
    ) * profile_penalty.get('Medicine', 1)

    # Computer Science
    if info_grade > 5:
        scores['Computer Science'] = difficulty_penalty * (
            0.3 * math_grade + 0.4 * info_grade + 0.002 * digital_skills + 0.1 * overall_grade
        ) * profile_penalty.get('Computer Science', 1)
    else:
        scores['Computer Science'] = difficulty_penalty * (
            0.6 * math_grade + 0.003 * digital_skills + 0.1 * overall_grade
        ) * profile_penalty.get('Computer Science', 1)

    # Mathematics
    scores['Mathematics'] = difficulty_penalty * (
        0.7 * math_grade + 0.2 * overall_grade + 0.1 * science_grade
    )

    # Physical Education & Sports
    if profile == 'Educație fizică și sport':
        scores['Physical Education'] = difficulty_penalty * (
            0.5 * 10 + 0.3 * overall_grade + 0.2 * communication_skills
        )
    else:
        scores['Physical Education'] = difficulty_penalty * (
            0.6 * sports_grade + 0.2 * overall_grade + 0.2 * communication_skills
        )

    # Technical Fields
    scores['Technical'] = difficulty_penalty * (
        0.5 * math_grade + 0.3 * science_grade + 0.002 * digital_skills
    ) * profile_penalty.get('Technical', 1)

    # Arts
    if profile == 'Artistic':
        scores['Arts'] = difficulty_penalty * (
            0.5 * 10 + 0.3 * language_grade + 0.2 * overall_grade
        ) * profile_penalty.get('Arts', 1)
    else:
        scores['Arts'] = difficulty_penalty * (
            0.6 * language_grade + 0.3 * communication_skills + 0.1 * overall_grade
        ) * profile_penalty.get('Arts', 1)

    # Recommend the best field
    recommended_field = max(scores, key=scores.get)
    return recommended_field, scores


def search_universities_near_city(city, interest):
    params = {
        "engine": "google",
        "q": f"universities near {city} Romania {interest}",
        "location": f"{city}, Romania",
        "num": 10,
        # "api_key": "9e0409085db46c894a9646b2d03041e2954fc586212772d70c2ab195e435dd69"
        "api_key": ""
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    universities = []
    keywords = [
        "university", "universitate", "faculty", "facultate", 
        "institute", "institut", "college", "colegiu", 
        "academia", "academie", "polytechnic", "politehnică", 
        "school of", "școală", "univ", "institute of"
    ]

    for result in results.get('organic_results', []):
        title = result.get('title', '').lower()
        link = result.get('link')
        if any(keyword in title for keyword in keywords):
            universities.append({'Title': result.get('title', ''), 'Link': link})
    return universities


def load_recommended_university_data(recommended_field):
    field_to_csv = {
        "Architecture": "./Admiteri CSV/ARHITECTURA_clean_2024.csv",
        "Law": "./Admiteri CSV/DREPT_clean_2024.csv",
        "Geography": "./Admiteri CSV/GEOGRAFIE_2024.csv",
        "Journalism": "./Admiteri CSV/JURNALISM_2024.csv",
        "Psychology": "./Admiteri CSV/PSIHOLOGIE_2024.csv",
        "Political Science": "./Admiteri CSV/SNSPA_2024.csv",
        "Medicine": "./Admiteri CSV/UMFCD_2019.csv",
        "Computer Science": "./Admiteri CSV/UNIBUC_CTI_INFO_2024.csv",
        "Mathematics": "./Admiteri CSV/UNIBUC_MATE_2024.csv",
        "Technical": "./Admiteri CSV/UPB_ACS_2014.csv",
        "Arts": "./Admiteri CSV/ARHITECTURA_clean_2024.csv",
        "Physical Education": None
    }

    csv_file = field_to_csv.get(recommended_field)
    if csv_file:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                return df
            except FileNotFoundError:
                print(f"File {csv_file} not found.")
                return None
        else:
            print(f"Path {csv_file} does not exist.")
            return None
    else:
        return None

def analyze_and_visualize(df, user_medie):
    """
    Plots the distribution of an admission score (like 'Medie BAC'),
    draws a vertical line for user_medie, then returns a figure as Base64 string.
    Also returns a short textual comparison to average.
    """
    possible_columns = [
        'Medie BAC',
        'Medie admitere (15% BAC +15% Lb.rom scris+ 70% examen scris)',
        'Medie admitere',
        'Nota',
        'Medie',
        'BAC'
    ]

    bac_column = None
    for col in possible_columns:
        if col in df.columns:
            bac_column = col
            break

    comparison_text = ""
    if df is not None and bac_column:
        # Plotting the distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[bac_column], bins=20, alpha=0.7, label='Students')
        ax.axvline(user_medie, color='red', linestyle='dashed', linewidth=2, label=f'User Medie: {user_medie}')
        ax.set_xlabel(bac_column)
        ax.set_ylabel('Number of Students')
        ax.set_title('Distribution of Admission Score Averages vs. User Score')
        ax.legend()
        ax.grid(True)

        # Convert figure to base64
        histogram_base64 = fig_to_base64(fig)

        # Compare user score to average
        avg_medie = df[bac_column].mean()
        comparison_text += f"Average Score in this field: {avg_medie:.2f}\n"
        if user_medie > avg_medie:
            comparison_text += "Your BAC score is above the average!\n"
        elif user_medie < avg_medie:
            comparison_text += "Your BAC score is below the average.\n"
        else:
            comparison_text += "Your BAC score is exactly at the average.\n"

        return histogram_base64, comparison_text
    else:
        # fallback if no data
        return "", "DataFrame is empty or 'Medie BAC' column is missing.\n"

def plot_comparison_bar_chart(df, user_data):
    """
    Creates a bar chart comparing user scores to dataset averages
    for specific numeric columns. Returns the figure as a Base64 string.
    """
    numeric_columns = ['NOTA_FINALA_EA', 'NOTA_FINALA_EC', 'NOTA_FINALA_ED', 'PUNCTAJ DIGITALE', 'Medie']

    # Ensure these columns exist and are numeric
    for col in numeric_columns:
        if col not in df.columns:
            # If any required column is missing, just return empty
            return "", "One or more numeric columns are missing in the dataset.\n"

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate dataset means
    dataset_means = df[numeric_columns].mean()

    # Create a comparison DataFrame
    comparison_df = pd.DataFrame({
        'User': [user_data.get(col, 0) for col in numeric_columns],
        'Dataset Average': dataset_means
    }, index=numeric_columns)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_title('Comparison of User Scores to Dataset Averages')
    ax.set_ylabel('Scores')
    ax.set_xlabel('Categories')
    plt.xticks(rotation=45)
    plt.tight_layout()

    base64_img = fig_to_base64(fig)
    text_output = "Analysis complete.\n"

    return base64_img, text_output


def full_recommendation(city, user_data):
    # 1. Calculate best field
    recommendation, all_scores = calculate_scores(user_data)

    # 2. Find relevant universities
    filtered_results = search_universities_near_city(city, recommendation)

    # 3. Load recommended university data
    df_recommended_uni = load_recommended_university_data(recommendation)

    histogram_image = ""
    radar_image = ""
    comparison_text = ""
    new_chart_image = ""

    if df_recommended_uni is not None and not df_recommended_uni.empty:
        histogram_image = plot_admission_histogram(df_recommended_uni, user_data['Medie'])

        # Some average logic for text
        possible_columns = ['Medie BAC', 'Medie admitere', 'Medie', 'BAC', 'Nota']
        bac_col = next((c for c in possible_columns if c in df_recommended_uni.columns), None)
        if bac_col:
            avg_medie = df_recommended_uni[bac_col].mean()
            comparison_text += f"Average Score in {recommendation}: {avg_medie:.2f}\n"
            if user_data['Medie'] > avg_medie:
                comparison_text += "Your BAC score is above the average!\n"
            elif user_data['Medie'] < avg_medie:
                comparison_text += "Your BAC score is below the average.\n"
            else:
                comparison_text += "Your BAC score is exactly the average.\n"
        else:
            comparison_text += "No valid column found in recommended CSV for comparison.\n"


        snippet_image, snippet_text = analyze_and_visualize(df_recommended_uni, user_data['Medie'])
        comparison_text += f"\n{snippet_text}"
        new_chart_image = snippet_image

    else:
        comparison_text += "No recommended CSV found for this field or CSV is empty.\n"

    # 4. Radar Chart
    try:
        df_bac = pd.read_csv("./BAC CSV/bac2019.csv")
        radar_image = plot_radar_chart(df_bac, user_data)

        # 4a. Generate the new bar chart from the same df_bac
        new_bar_chart_img, bar_chart_text = plot_comparison_bar_chart(df_bac, user_data)
        comparison_text += f"\n{bar_chart_text}"

    except FileNotFoundError:
        comparison_text += "\nCould not find bac2019.csv for radar/bar charts.\n"

    return (
        recommendation,
        all_scores,
        filtered_results,
        comparison_text,
        histogram_image,
        radar_image,
        new_bar_chart_img
    )


def plot_admission_histogram(df, user_medie):
    """
    Generates a histogram of "Medie BAC" (or a similar column in df),
    draws a red line for user_medie, returns the figure as Base64.
    """
    possible_columns = [
        'Medie BAC',
        'Medie admitere (15% BAC +15% Lb.rom scris+ 70% examen scris)',
        'Medie admitere', 'Nota', 'Medie', 'BAC'
    ]
    bac_column = None
    for col in possible_columns:
        if col in df.columns:
            bac_column = col
            break

    if bac_column is None:
        # If the column is not found, return an empty string or a placeholder
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[bac_column], bins=20, alpha=0.7, color='blue', label='Admission Scores')
    ax.axvline(user_medie, color='red', linestyle='dashed', linewidth=2, label=f'User Medie: {user_medie}')
    ax.set_xlabel(bac_column)
    ax.set_ylabel('Number of Students')
    ax.set_title('Distribution of Admission Scores vs. User Score')
    ax.legend()
    ax.grid(True)

    return fig_to_base64(fig)

def plot_radar_chart(df_bac, user_data):
    """
    Creates a radar chart comparing user’s numeric fields with the dataset average.
    Returns the figure as Base64.
    """
    numeric_columns = ['NOTA_FINALA_EA', 'NOTA_FINALA_EC', 'NOTA_FINALA_ED', 'PUNCTAJ DIGITALE', 'Medie']

    # Ensure these columns exist and are numeric in df_bac
    for col in numeric_columns:
        if col not in df_bac.columns:
            return ""

    df_bac[numeric_columns] = df_bac[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df_bac['PUNCTAJ DIGITALE'] = df_bac['PUNCTAJ DIGITALE'] / 10.0

    # Dataset average
    dataset_means = df_bac[numeric_columns].mean()

    # Prepare data for the radar chart
    labels = numeric_columns
    user_stats = [user_data.get(col, 0) for col in numeric_columns]
    avg_stats = [dataset_means[col] for col in numeric_columns]

    # Radar chart requires the angles for each variable
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    user_stats += user_stats[:1]      # repeat first value to close the circle
    avg_stats += avg_stats[:1]        # repeat first value to close the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot user data
    ax.plot(angles, user_stats, 'o-', linewidth=2, label='User')
    ax.fill(angles, user_stats, alpha=0.25)

    # Plot average data
    ax.plot(angles, avg_stats, 'o-', linewidth=2, label='Dataset Average')
    ax.fill(angles, avg_stats, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Radar Chart: User vs. Dataset Average')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    return fig_to_base64(fig)

# ==============================================================
# =============== Flask Routes & Template Usage ================
# ==============================================================

@app.route("/", methods=["GET", "POST"])
def home_page():

    city_val = "Bucharest"
    profil_val = "Real"
    specializare_val = "Matematica-Informatica"
    subiect_ea_val = "Limba română (REAL)"
    subiect_ec_val = ""
    subiect_ed_val = ""
    nota_finala_ea_val = ""
    nota_finala_ec_val = ""
    nota_finala_ed_val = ""
    punctaj_dig_val = ""
    oral_pmo_val = ""
    medie_val = ""
    recommendation = None
    sorted_scores = []
    universities = []
    comparison_text = ""
    histogram_img = ""
    radar_img = ""
    new_chart_img = ""

    if request.method == "POST":
        # The user just submitted the form -> do all the recommendation logic

        city_val = request.form.get("city", "")
        profil_val = request.form.get("Profil", "")
        specializare_val = request.form.get("Specializare", "")
        subiect_ea_val = request.form.get("Subiect_ea", "")
        subiect_ec_val = request.form.get("Subiect_ec", "")
        subiect_ed_val = request.form.get("Subiect_ed", "")
        nota_finala_ea_val = float(request.form.get("NOTA_FINALA_EA", 0))
        nota_finala_ec_val = float(request.form.get("NOTA_FINALA_EC", 0))
        nota_finala_ed_val = float(request.form.get("NOTA_FINALA_ED", 0))
        punctaj_dig_val = float(request.form.get("PUNCTAJ_DIGITALE", 0))
        oral_pmo_val = float(request.form.get("ORAL_PMO", 0))
        medie_val = float(request.form.get("Medie", 0))

        city = request.form.get("city", "Bucharest")
        user_data = {
            'Profil': request.form.get('Profil', ''),
            'Specializare': request.form.get('Specializare', ''),
            'Subiect ea': request.form.get('Subiect_ea', ''),
            'Subiect ec': request.form.get('Subiect_ec', ''),
            'Subiect ed': request.form.get('Subiect_ed', ''),
            'NOTA_FINALA_EA': float(request.form.get('NOTA_FINALA_EA', 0)),
            'NOTA_FINALA_EC': float(request.form.get('NOTA_FINALA_EC', 0)),
            'NOTA_FINALA_ED': float(request.form.get('NOTA_FINALA_ED', 0)),
            'PUNCTAJ DIGITALE': float(request.form.get('PUNCTAJ_DIGITALE', 0)),
            'ORAL_PMO': float(request.form.get('ORAL_PMO', 0)),
            'Medie': float(request.form.get('Medie', 0))
        }

        recommendation, all_scores, universities, comparison_text,new_chart_img, histogram_img, radar_img = full_recommendation(city, user_data)

        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return render_template(
            "index.html",
            city=city,
            profil=profil_val,
            specializare=specializare_val,
            subiect_ea=subiect_ea_val,
            subiect_ec_val = subiect_ec_val,
            subiect_ed = subiect_ed_val,
            nota_finala_ea = nota_finala_ea_val,
            nota_finala_ec = nota_finala_ec_val,
            nota_finala_ed = nota_finala_ed_val,
            punctaj_dig = punctaj_dig_val,
            oral_pmo = oral_pmo_val,
            medie = medie_val,
            recommendation=recommendation,
            sorted_scores=sorted_scores,
            universities=universities,
            comparison_text=comparison_text,
            new_chart_img=new_chart_img,
            histogram_img=histogram_img,
            radar_img=radar_img
        )
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
