import pandas as pd
from pathlib import Path
def pedir_caracteristicas():
    caracteristicas = {
        "Marital status": [2],
        "Application mode": [39],
        "Application order": [1],
        "Course": [8014],
        "Daytime/evening attendance\t": [0],
        "Previous qualification": [1],
        "Previous qualification (grade)": [145],
        "Nacionality": [1],
        "Mother's qualification": [37],
        "Father's qualification": [37],
        "Mother's occupation": [9],
        "Father's occupation": [6],
        "Admission grade": [138.5],
        "Displaced": [0],
        "Educational special needs": [0],
        "Debtor": [0],
        "Tuition fees up to date": [1],
        "Gender": [1],
        "Scholarship holder": [0],
        "Age at enrollment": [22],
        "International": [0],
        "Curricular units 1st sem (credited)": [0],
        "Curricular units 1st sem (enrolled)": [6],
        "Curricular units 1st sem (evaluations)": [9],
        "Curricular units 1st sem (approved)": [7],
        "Curricular units 1st sem (grade)": [14.5],
        "Curricular units 1st sem (without evaluations)": [0],
        "Curricular units 2nd sem (credited)": [0],
        "Curricular units 2nd sem (enrolled)": [6],
        "Curricular units 2nd sem (evaluations)": [10],
        "Curricular units 2nd sem (approved)": [6],
        "Curricular units 2nd sem (grade)": [13.8],
        "Curricular units 2nd sem (without evaluations)": [0],
        "Unemployment rate": [8.9],
        "Inflation rate": [1.4],
        "GDP": [3.51]
    }
    return pd.DataFrame(caracteristicas, index=[0])

def return_info_text():
    current_dir = Path(__file__).resolve().parent
    info_text_path = current_dir / 'info_text.txt'
    with open(info_text_path, 'r', encoding='utf-8') as file:
        info_text = file.read()

    return info_text

def return_info_experto():
    #info_experto = "I think the second-quarter subjects are slightly more complex."
    info_experto = "The student failed more subjects in the second semester because he was going through a difficult family situation due to the loss of a close relative. This situation affected him emotionally, causing him to be more absent, distracted, and less focused in class, which could explain the decline in his academic performance during that period."
    return info_experto