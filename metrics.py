from prometheus_client import Counter, Histogram
jobs_enqueued_total = Counter("jobs_enqueued_total","Jobs enqueued")
jobs_completed_total = Counter("jobs_completed_total","Jobs completed")
page_latency_ms_by_template = Histogram("page_latency_ms_by_template","OCR/PP page latency",
                                        ["step","template"], buckets=(10,50,100,250,500,1000,2000,5000,10000))

def observe_page_latency(step: str, ms: int, template: str):
    try:
        page_latency_ms_by_template.labels(step=step, template=template).observe(ms)
    except Exception:
        pass
