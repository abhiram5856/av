from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

from backend.core.limiter import limiter
from datetime import datetime
import hashlib
import random

# In-memory mock database for subscribers and deduplication
subscribers = []
processed_alerts = set()
alert_statuses = {}

class AlertSubscription(BaseModel):
    phone_number: str
    region: str
    crop: str

class OutbreakAlert(BaseModel):
    disease: str
    region: str
    severity: str
    message: str

# In-memory mock database for subscribers
subscribers = []

@router.post("/subscribe")
@limiter.limit("10/minute")
def subscribe_to_alerts(request: Request, subscription: AlertSubscription):
    # Mock adding to DB
    subscribers.append(subscription)
    logger.info(f"New SMS subscription for {subscription.phone_number} in {subscription.region} for {subscription.crop}")
    return {"status": "success", "message": f"Successfully subscribed {subscription.phone_number} to outbreak alerts."}

def mock_send_twilio_sms(phone: str, msg: str, alert_id: str):
    # Simulate API call with a chance of failure
    logger.info(f"Processing SMS to {phone} for alert {alert_id}")
    
    # 10% chance of failure for realism in QA
    is_success = random.random() > 0.1
    
    if is_success:
        alert_statuses[alert_id] = "SENT"
        logger.info(f"[TWILIO MOCK] SUCCESSFULLY sent SMS to {phone}:\n{msg}")
        return True
    else:
        alert_statuses[alert_id] = "FAILED"
        logger.error(f"[TWILIO MOCK] FAILED to send SMS to {phone}")
        return False

@router.post("/trigger_outbreak")
@limiter.limit("5/minute")
def trigger_outbreak_alert(request: Request, alert: OutbreakAlert, background_tasks: BackgroundTasks):
    """
    Called by the analytics engine when a regional threshold is crossed.
    Broadcasts SMS to all subscribed farmers in the region.
    """
    # Deduplication key based on disease, region, and severity
    # Add a time component (e.g., today's date) so we don't alert twice in one day for the same event
    today = datetime.utcnow().strftime("%Y-%m-%d")
    dedup_string = f"{alert.disease}_{alert.region}_{alert.severity}_{today}".encode('utf-8')
    event_hash = hashlib.md5(dedup_string).hexdigest()
    
    if event_hash in processed_alerts:
        return {"status": "skipped", "message": "Duplicate alert prevented. Already triggered today."}
    affected_subs = [s for s in subscribers if s.region.lower() == alert.region.lower()]
    
    if not affected_subs:
        return {"status": "no_subscribers", "message": "No subscribers in this region."}

    processed_alerts.add(event_hash)
    
    for sub in affected_subs:
        sms_body = f"AGRIVISION ALERT: High risk of {alert.disease} in {alert.region} ({alert.severity}). {alert.message}"
        alert_id = f"{event_hash}_{sub.phone_number}"
        alert_statuses[alert_id] = "QUEUED"
        background_tasks.add_task(mock_send_twilio_sms, sub.phone_number, sms_body, alert_id)

    return {"status": "alerts_queued", "count": len(affected_subs), "event_id": event_hash}
