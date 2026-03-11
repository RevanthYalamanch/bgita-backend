# backend/auth_logic.py
CLINICIAN_KEY = "GITA_MED_2026"

def verify_clinician(email: str, code: str):
    """
    Validates if a clinician is authorized to access 
    clinical dashboards based on the special code.
    """
    if code == CLINICIAN_KEY:
        return {"status": "authorized", "role": "psychiatrist"}
    return {"status": "unauthorized", "role": "user"}