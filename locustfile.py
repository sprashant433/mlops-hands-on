from locust import HttpUser, between, task


class LoanApprovalUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        payload = {
            "age": 35,
            "income": 75000,
            "loan_amount": 25000,
            "credit_score": 700,
            "employment_years": 5,
            "debt_to_income": 0.3,
        }

        self.client.post(
            "/predict",
            json=payload,
            headers={"x-request-id": "locust-load-test"},
        )

    @task(1)
    def health(self):
        self.client.get("/health")