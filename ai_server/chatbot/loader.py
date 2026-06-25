import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SURVEY_FILE = os.path.join(DATA_DIR, "dummy_surveys.json")
QUICK_REPLIES_FILE = os.path.join(DATA_DIR, "quick_replies.json")
INTENT_EXAMPLES_FILE = os.path.join(DATA_DIR, "intent_examples.json")
BUSAN_YOUTH_POLICY_FILE = os.path.join(DATA_DIR, "부산광역시_청년지원정책 현황.csv")
BUSAN_JOB_SERVICE_FILE = os.path.join(DATA_DIR, "청년일자리지원 서비스.csv")

EXCLUDED_FILES = ["dummy_surveys.json", "quick_replies.json", "intent_examples.json"]


def load_intent_examples():
    with open(INTENT_EXAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories", [])


def load_all_jobs():
    all_jobs = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue
        if filename in EXCLUDED_FILES:
            continue
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        category = data.get("category", "")
        for job in data.get("jobs", []):
            job_category_mid = job.get("position", {}).get("job_category", {}).get("mid", "")
            job["category"] = category or job_category_mid
            all_jobs.append(job)
    return all_jobs


def load_trend_summaries():
    trends = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json"):
            continue
        if filename in EXCLUDED_FILES:
            continue
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        trend = data.get("trend_summary", {})
        category = data.get("category", "")
        if trend:
            trends.append({
                "category": category,
                "hot_frameworks": trend.get("hot_frameworks", []),
                "hot_languages": trend.get("hot_languages", []),
                "hot_tools": trend.get("hot_tools", []),
                "talent_keywords": trend.get("talent_keywords", []),
                "salary_trend": trend.get("salary_trend", ""),
                "welfare_trend": trend.get("welfare_trend", ""),
            })
    return trends


def load_user_surveys():
    with open(SURVEY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_user_info(user_id: str):
    surveys = load_user_surveys()
    for survey in surveys:
        if survey.get("user_id") == user_id:
            return survey
    return None


def load_quick_replies():
    with open(QUICK_REPLIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("quick_replies", [])


def get_category_stats():
    all_jobs = load_all_jobs()
    stats = {}
    for job in all_jobs:
        category = job.get("category", "")
        if category not in stats:
            stats[category] = {"count": 0, "view_count": 0, "apply_count": 0, "bookmark_count": 0}
        stats[category]["count"] += 1
        stats[category]["view_count"] += job.get("stats", {}).get("view_count", 0)
        stats[category]["apply_count"] += job.get("stats", {}).get("apply_count", 0)
        stats[category]["bookmark_count"] += job.get("stats", {}).get("bookmark_count", 0)
    return stats


def load_csv_records(file_path: str):
    import csv
    if not os.path.exists(file_path):
        print(f"[WARN] CSV 파일을 찾을 수 없습니다: {file_path}")
        return []
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            with open(file_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                records = []
                for row in reader:
                    cleaned_row = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        clean_key = key.strip()
                        clean_value = value.strip() if isinstance(value, str) else value
                        cleaned_row[clean_key] = clean_value or ""
                    records.append(cleaned_row)
                print(f"[LOAD] CSV 로드 완료: {os.path.basename(file_path)} ({len(records)}건)")
                return records
        except UnicodeDecodeError:
            continue
    print(f"[ERROR] CSV 인코딩을 확인해주세요: {file_path}")
    return []


def load_busan_youth_policies():
    return load_csv_records(BUSAN_YOUTH_POLICY_FILE)


def load_busan_job_services():
    return load_csv_records(BUSAN_JOB_SERVICE_FILE)
