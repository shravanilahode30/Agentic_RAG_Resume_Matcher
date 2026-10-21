import time
import statistics
from tools import search_resume, score_jd_match

# 10 structurally different JDs to test generalization, not just one lucky case
test_jds = [
    ("Backend - FastAPI", "We need a Python/FastAPI developer with PostgreSQL, Redis, and Docker experience."),
    ("Backend - Django", "Looking for a Django REST Framework developer with PostgreSQL and REST API experience."),
    ("Frontend - React", "Frontend Developer needed with React, JavaScript, HTML, CSS, and REST API integration."),
    ("Frontend - Vue", "Seeking a Vue.js developer familiar with component-based architecture and JavaScript."),
    ("Data - Analytics", "Data Analyst role requiring SQL, Python, Pandas, and dashboard-building experience."),
    ("Data - ML", "Machine Learning Engineer needed with Python, scikit-learn, and model deployment experience."),
    ("Full Stack", "Full Stack Developer with Django, React, PostgreSQL, and REST API experience required."),
    ("QA/Testing", "QA Engineer needed with Selenium, PyTest, and automated testing framework experience."),
    ("DevOps", "DevOps Engineer with Docker, Kubernetes, CI/CD pipelines, and AWS experience required."),
    ("Database", "Database Developer needed with strong SQL, PostgreSQL, query optimization, and indexing skills."),
]

search_times = []
score_times = []
total_times = []

print(f"{'JD Type':<20} {'Search (s)':<12} {'Score (s)':<12} {'Total (s)':<12}")
print("-" * 56)

for label, jd_text in test_jds:
    start_total = time.perf_counter()

    start_search = time.perf_counter()
    context = search_resume(jd_text, n_results=2)
    search_elapsed = time.perf_counter() - start_search

    start_score = time.perf_counter()
    result = score_jd_match(jd_text, context)
    score_elapsed = time.perf_counter() - start_score

    total_elapsed = time.perf_counter() - start_total

    search_times.append(search_elapsed)
    score_times.append(score_elapsed)
    total_times.append(total_elapsed)

    status = "OK" if "error" not in result else "ERROR"
    print(f"{label:<20} {search_elapsed:<12.3f} {score_elapsed:<12.3f} {total_elapsed:<12.3f} [{status}]")

print("-" * 56)
print(f"\nAverage search_resume latency: {statistics.mean(search_times):.3f}s")
print(f"Average score_jd_match latency: {statistics.mean(score_times):.3f}s")
print(f"Average total (search + score) latency: {statistics.mean(total_times):.3f}s")
print(f"Min total: {min(total_times):.3f}s  |  Max total: {max(total_times):.3f}s")