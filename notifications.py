"""
Notification utility functions for HireArchy
Standalone module - no Flask app needed here
"""

import logging
from datetime import datetime
from bson.objectid import ObjectId
from database import mongo

logger = logging.getLogger(__name__)

# Don't initialize collection at import time - use lazy initialization
def get_notifications_collection():
    """Get NOTIFICATIONS collection with lazy initialization"""
    try:
        return mongo.db.NOTIFICATIONS
    except Exception as e:
        logger.error(f"❌ Failed to get NOTIFICATIONS collection: {e}")
        return None


def create_notification(user_id, title, message, notification_type, related_id=None, action_url=None):
    """Create a new notification"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        logger.error("NOTIFICATIONS collection not available")
        return None
    
    try:
        notification_data = {
            "user_id": ObjectId(user_id),
            "title": title,
            "message": message,
            "type": notification_type,
            "related_id": ObjectId(related_id) if related_id else None,
            "action_url": action_url,
            "is_read": False,
            "created_at": datetime.now(),
            "is_active": True
        }
        
        result = NOTIFICATIONS.insert_one(notification_data)
        logger.info(f"✅ Notification created for user {user_id}: {title}")
        return result.inserted_id
        
    except Exception as e:
        logger.error(f"❌ Error creating notification: {e}")
        return None


def create_application_notification(applicant_id, hr_id, job_id, job_title, applicant_name):
    """Create notifications when someone applies to a job"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        logger.error("❌ Notifications disabled - collection not available")
        return
    
    try:
        # Notification for applicant
        create_notification(
            user_id=applicant_id,
            title="Application Submitted",
            message=f"Your application for {job_title} has been submitted successfully.",
            notification_type="application_submitted",
            related_id=job_id,
            action_url=f"/application_status/{job_id}"
        )
        
        # Notification for HR
        create_notification(
            user_id=hr_id,
            title="New Application Received",
            message=f"{applicant_name} applied for {job_title}",
            notification_type="new_application",
            related_id=job_id,
            action_url=f"/HR1/Company_Candidates?job_id={job_id}"
        )
        
        logger.info(f"✅ Application notifications created for job: {job_title}")
        
    except Exception as e:
        logger.error(f"❌ Error creating application notifications: {e}")


def create_interview_notification(applicant_id, hr_id, job_title, interview_date, interview_type):
    """Create notifications for interview scheduling"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        logger.error("❌ Notifications disabled - collection not available")
        return
    
    try:
        # Format interview date
        if isinstance(interview_date, datetime):
            date_str = interview_date.strftime('%B %d, %Y at %I:%M %p')
        else:
            date_str = str(interview_date)
        
        # Notification for applicant
        create_notification(
            user_id=applicant_id,
            title="Interview Scheduled",
            message=f"Your interview for {job_title} is scheduled for {date_str}",
            notification_type="interview_scheduled",
            related_id=None,
            action_url="/interviews"
        )
        
        # Notification for HR
        create_notification(
            user_id=hr_id,
            title="Interview Scheduled",
            message=f"Interview scheduled for {job_title} on {date_str}",
            notification_type="interview_scheduled",
            related_id=None,
            action_url="/HR1/interviews"
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating interview notifications: {e}")


def create_status_update_notification(applicant_id, job_title, old_status, new_status, job_id):
    """Create notification when application status changes"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        logger.error("❌ Notifications disabled - collection not available")
        return
    
    try:
        status_messages = {
            'shortlisted': f"Great news! You've been shortlisted for {job_title}",
            'interview_pending': f"Your application for {job_title} is moving to the interview stage",
            'interviewed': f"Thank you for interviewing for {job_title}. We'll be in touch soon",
            'hired': f"Congratulations! You've been selected for {job_title}",
            'not_selected': f"Thank you for your interest in {job_title}. We've decided to move forward with other candidates"
        }
        
        message = status_messages.get(new_status, f"Your application status for {job_title} has been updated to {new_status}")
        
        create_notification(
            user_id=applicant_id,
            title="Application Status Update",
            message=message,
            notification_type="status_update",
            related_id=job_id,
            action_url=f"/application_status/{job_id}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating status update notification: {e}")


def get_user_notifications(user_id, limit=20, unread_only=False):
    """Get notifications for a user"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        logger.warning("NOTIFICATIONS collection not available")
        return []
    
    try:
        query = {"user_id": ObjectId(user_id), "is_active": True}
        if unread_only:
            query["is_read"] = False
        
        notifications = list(NOTIFICATIONS.find(query)
                           .sort("created_at", -1)
                           .limit(limit))
        
        return notifications
        
    except Exception as e:
        logger.error(f"❌ Error fetching notifications for user {user_id}: {e}")
        return []


def get_unread_count(user_id):
    """Get count of unread notifications"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        return 0
    
    try:
        count = NOTIFICATIONS.count_documents({
            "user_id": ObjectId(user_id),
            "is_read": False,
            "is_active": True
        })
        return count
    except Exception as e:
        logger.error(f"❌ Error getting unread count for user {user_id}: {e}")
        return 0


def mark_notification_read(notification_id, user_id):
    """Mark a notification as read"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        return False
    
    try:
        result = NOTIFICATIONS.update_one(
            {
                "_id": ObjectId(notification_id),
                "user_id": ObjectId(user_id)
            },
            {"$set": {"is_read": True, "read_at": datetime.now()}}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Error marking notification as read: {e}")
        return False


def mark_all_notifications_read(user_id):
    """Mark all notifications as read for a user"""
    NOTIFICATIONS = get_notifications_collection()
    if NOTIFICATIONS is None:
        return 0
    
    try:
        result = NOTIFICATIONS.update_many(
            {
                "user_id": ObjectId(user_id),
                "is_read": False
            },
            {"$set": {"is_read": True, "read_at": datetime.now()}}
        )
        return result.modified_count
    except Exception as e:
        logger.error(f"❌ Error marking all notifications as read: {e}")
        return 0