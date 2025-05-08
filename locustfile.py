from locust import HttpUser, task, between

class StressTestUser(HttpUser):
    wait_time = between(1, 3)  # simulate real users with small wait

    @task
    def predict_heart_disease(self):
        payload = {
            "age": 60,
            "sex": 1,
            "chest_pain_type": 2,
            "resting_blood_pressure": 140,
            "cholestoral": 289,
            "fasting_blood_sugar": 0,
            "rest_ecg": 1,
            "Max_heart_rate": 172,
            "exercise_induced_angina": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "vessels_colored_by_flourosopy": 0,
            "thalassemia": 2
        }

        self.client.post("/predict", json=payload)
