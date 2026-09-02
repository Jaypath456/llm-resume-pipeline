from tailor_resumes import check_unsupported_terms, flag_unverified_terms
master = open("Master_Resume_Content.md").read()
fake_content = {
    "experience": {},
    "skills": {
        "Software Engineering": "REST API, ETL Pipelines, CI/CD Pipelines, Dynamic SQL Querying, Microservices Architecture"
    },
    "academic_projects": {}
}
print("UNSUPPORTED:", check_unsupported_terms(fake_content, master))
print("UNVERIFIED:", flag_unverified_terms(fake_content, master))

