from locustfile import LoanApprovalUser


def test_loan_approval_user_has_tasks():
    tasks = LoanApprovalUser.tasks

    assert len(tasks) > 0
