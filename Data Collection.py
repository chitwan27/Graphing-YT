import os
import sys
import sqlite3
import time
import requests
import datetime
import json
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# ======= CONFIG =======
API_KEY = None  # Not used in OAuth
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

# Get absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(SCRIPT_DIR, 'secrets', 'token.json')
CLIENT_SECRETS_FILE = os.path.join(SCRIPT_DIR, 'secrets', 'client_secrets.json')
DB_FILE = os.path.join(SCRIPT_DIR, 'youtube_graph.db')
LOG_FILE = os.path.join(SCRIPT_DIR, 'crawler.log')

BASE_URL = 'https://www.googleapis.com/youtube/v3/subscriptions'
DAILY_QUOTA_LIMIT = 10000

# ======= LOGGING SETUP =======
def setup_logging():
    """Setup logging for cron job compatibility"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ======= AUTH =======
def get_credentials():
    creds = None
    try:
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                from google_auth_oauthlib.flow import InstalledAppFlow
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            # Ensure secrets directory exists
            os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        return creds.token
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise

ACCESS_TOKEN = get_credentials()
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# ======= STATS TRACKING =======
def get_channel_counts():
    """Get current channel counts for statistics"""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM channels')
        total_channels = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM channels WHERE last_crawled IS NOT NULL')
        crawled_channels = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM subscriptions')
        total_subscriptions = c.fetchone()[0]
        
        return total_channels, crawled_channels, total_subscriptions

# ======= QUOTA TRACKING =======
class QuotaTracker:
    def __init__(self):
        self.quota_used = 0
        self.quota_limit = DAILY_QUOTA_LIMIT
        
    def use_quota(self, cost=1):
        """Use quota and return True if we can continue, False if limit reached"""
        self.quota_used += cost
        logger.info(f"Quota used: {self.quota_used}/{self.quota_limit}")
        return self.quota_used < self.quota_limit
    
    def can_make_request(self, cost=1):
        """Check if we can make a request without exceeding quota"""
        return (self.quota_used + cost) <= self.quota_limit

# ======= DB INIT =======
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                last_crawled TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS subscriptions (
                from_channel TEXT,
                to_channel TEXT,
                crawl_time TEXT,
                PRIMARY KEY (from_channel, to_channel)
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS channel_names (
                channel_id TEXT PRIMARY KEY,
                title TEXT
            )
        ''')
        conn.commit()

# ======= API CALLS =======
def parse_subscriptions_minimal(data):
    results = []
    for item in data.get("items", []):
        try:
            to_channel_id = item['snippet']['resourceId']['channelId']
            title = item['snippet']['title']
            results.append((to_channel_id, title))
        except KeyError:
            continue
    return results

def get_subscriptions(channel_id, quota_tracker):
    """
    Fetch public subscriptions. Return (list of (to_channel_id, title), completed_successfully).
    completed_successfully is True if we got all subscriptions without hitting quota OR if subscriptions are private.
    """
    results = []
    params = {
        'part': 'snippet',
        'channelId': channel_id,
        'maxResults': 50
    }

    try:
        while True:
            # Check if we can make the request (cost = 1 for subscriptions.list)
            if not quota_tracker.can_make_request(1):
                logger.warning(f"Quota limit would be exceeded, stopping subscription fetch for {channel_id}")
                return results, False
            
            res = requests.get(BASE_URL, headers=HEADERS, params=params)
            
            # Use quota after making the request
            if not quota_tracker.use_quota(1):
                logger.warning(f"Quota limit reached after request for {channel_id}")
                # We got the response but hit quota, so we can process this response
                # but should not continue with more requests
                if res.status_code == 200:
                    data = res.json()
                    results += parse_subscriptions_minimal(data)
                return results, False
            
            if res.status_code != 200:
                # Check if this is a private subscriptions error (403 subscriptionForbidden)
                if res.status_code == 403:
                    try:
                        error_data = res.json()
                        if (error_data.get('error', {}).get('errors', [{}])[0].get('reason') == 'subscriptionForbidden'):
                            logger.info(f"Channel {channel_id} has private subscriptions")
                            return results, True  # Mark as completed since this is permanent
                    except:
                        pass  # If we can't parse the error, fall through to general error handling
                
                # For any other error, we should still mark as completed to avoid infinite retries
                # These could be: channel deleted, suspended, or other permanent API errors
                logger.warning(f"Error {res.status_code} for {channel_id}: {res.text}")
                return results, True  # Mark as completed to move on from this channel

            data = res.json()
            results += parse_subscriptions_minimal(data)

            if 'nextPageToken' in data:
                params['pageToken'] = data['nextPageToken']
                time.sleep(0.1)
            else:
                # Successfully got all subscriptions
                return results, True
                
    except Exception as e:
        logger.error(f"Exception for {channel_id}: {e}")
        return results, False

# ======= MAIN CRAWLER =======
def crawl():
    init_db()
    now = datetime.datetime.utcnow().isoformat()
    quota_tracker = QuotaTracker()
    
    # Get initial stats
    initial_total, initial_crawled, initial_subs = get_channel_counts()
    
    logger.info(f"Starting crawl with quota limit: {DAILY_QUOTA_LIMIT}")
    logger.info(f"Initial stats - Total channels: {initial_total}, Crawled: {initial_crawled}, Subscriptions: {initial_subs}")

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()

        channels_crawled = 0
        
        while quota_tracker.can_make_request(1):  # Need at least 1 quota unit to make a request
            # Step 1: Get 1 uncrawled or stale channel
            c.execute('''
                SELECT channel_id FROM channels
                WHERE last_crawled IS NULL OR last_crawled < ?
                ORDER BY last_crawled ASC
                LIMIT 1
            ''', ((datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat(),))

            row = c.fetchone()
            if not row:
                logger.info("No more channels to crawl.")
                break
                
            channel_id = row[0]
            logger.info(f"Crawling channel {channels_crawled + 1}: {channel_id}")

            # Step 2: Get their subscriptions
            subscriptions, completed_successfully = get_subscriptions(channel_id, quota_tracker)
            logger.info(f"Found {len(subscriptions)} subscriptions. Complete: {completed_successfully}")

            # Step 3: Store the subscriptions we got
            for to_channel_id, title in subscriptions:
                try:
                    # Insert into subscriptions
                    c.execute('''
                        INSERT OR IGNORE INTO subscriptions (from_channel, to_channel, crawl_time)
                        VALUES (?, ?, ?)
                    ''', (channel_id, to_channel_id, now))

                    # Insert into channels to crawl later
                    c.execute('''
                        INSERT OR IGNORE INTO channels (channel_id, last_crawled)
                        VALUES (?, NULL)
                    ''', (to_channel_id,))

                    # Store title
                    c.execute('''
                        INSERT OR IGNORE INTO channel_names (channel_id, title)
                        VALUES (?, ?)
                    ''', (to_channel_id, title))
                except Exception as e:
                    logger.error(f"DB insert error for {to_channel_id}: {e}")

            # Step 4: Mark channel as crawled ONLY if we got all subscriptions
            if completed_successfully:
                c.execute('UPDATE channels SET last_crawled = ? WHERE channel_id = ?', (now, channel_id))
                logger.info(f"Channel {channel_id} marked as fully crawled")
            else:
                logger.warning(f"Channel {channel_id} NOT marked as crawled (incomplete due to quota/error)")

            conn.commit()
            channels_crawled += 1
            
            # If we didn't complete successfully due to quota, break out of the loop
            if not completed_successfully and not quota_tracker.can_make_request(1):
                logger.info("Quota limit reached, stopping crawl")
                break

    # Get final stats and log summary
    final_total, final_crawled, final_subs = get_channel_counts()
    new_channels = final_total - initial_total
    new_subscriptions = final_subs - initial_subs
    
    logger.info(f"Crawl completed. Channels processed: {channels_crawled}, Quota used: {quota_tracker.quota_used}/{quota_tracker.quota_limit}")
    logger.info(f"New channels discovered: {new_channels}, New subscriptions: {new_subscriptions}")
    logger.info(f"Final stats - Total channels: {final_total}, Crawled: {final_crawled}, Subscriptions: {final_subs}")

# ======= ENTRY POINT =======
if __name__ == "__main__":
    try:
        # Change to script directory for cron compatibility
        os.chdir(SCRIPT_DIR)
        crawl()
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)