import csv
import random

rows = []

for i in range(200):
    age = random.randint(40, 90)
    sex = random.choice(["male", "female"])
    prior_admissions = random.randint(0, 6)
    length_of_stay = random.randint(1, 14)
    comorbidity = random.randint(0, 8)
    diabetes = random.choice(["true", "false"])
    hypertension = random.choice(["true", "false"])
    discharge = random.choice(["home", "rehab", "skilled_nursing", "home_health"])
    follow_up = random.choice(["true", "false"])
    med_risk = random.choice(["low", "medium", "high"])

    # better variability logic
    risk_score = (
        prior_admissions * 0.4 +
        comorbidity * 0.3 +
        (med_risk == "high") * 1.5 +
        (follow_up == "false") * 1 +
        random.random() * 2
    )

    readmitted = 1 if risk_score > 3 else 0

    rows.append([
        age, sex, prior_admissions, length_of_stay,
        comorbidity, diabetes, hypertension,
        discharge, follow_up, med_risk, readmitted
    ])

with open("readmission_data_large.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "age", "sex", "prior_admissions_12m", "length_of_last_stay",
        "comorbidity_count", "diabetes", "hypertension",
        "discharge_disposition", "follow_up_scheduled",
        "medication_adherence_risk", "readmitted"
    ])
    writer.writerows(rows)

print("New dataset generated!")