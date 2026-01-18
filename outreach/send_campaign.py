#!/usr/bin/env python3
"""
SurrAPI Email Outreach Campaign
===============================

Automated email sequence sender for CFD prospects.
Uses Gmail SMTP or SendGrid for delivery.

Usage:
    python outreach/send_campaign.py --csv sales/leads/cfd_prospects_50.csv
    python outreach/send_campaign.py --dry-run  # Preview without sending
    
Environment Variables:
    SURRAPI_EMAIL_FROM - Sender email
    SURRAPI_EMAIL_PASSWORD - App password (Gmail) or API key (SendGrid)
    SURRAPI_EMAIL_PROVIDER - 'gmail' or 'sendgrid'
"""

import os
import sys
import csv
import time
import smtplib
import argparse
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Dict, Optional
import json

# =============================================================================
# Configuration
# =============================================================================

EMAIL_FROM = os.getenv("SURRAPI_EMAIL_FROM", "team@surrapi.io")
EMAIL_PASSWORD = os.getenv("SURRAPI_EMAIL_PASSWORD", "")
EMAIL_PROVIDER = os.getenv("SURRAPI_EMAIL_PROVIDER", "gmail")

# Rate limiting
MAX_EMAILS_PER_DAY = 30
DELAY_BETWEEN_EMAILS = 60  # seconds

# Email templates
TEMPLATES = {
    "touch_1": {
        "subject": "300 ms CFD instead of 6 h queue?",
        "body": """Hi {first_name},

Running {tech_stack} burns ~$120 and half a day every design loop.

We pre-trained an FNO on 15 TB of validated turbulence data. Now you get velocity/pressure fields in 300 ms through a REST call.

Still costs $0.25 but saves 30-300 core-hours.

Swagger demo: https://demo.surrapi.io
Worth 15 min next week?

Best,
Adam
Founder, SurrAPI"""
    },
    
    "touch_2": {
        "subject": "Re: 300 ms CFD – accuracy data attached",
        "body": """Hi {first_name},

Quick follow-up with some numbers:

• Drag coefficient error: <1% vs full Fluent solve
• Reynolds range: 500 - 10,000
• Mach range: 0.05 - 0.6

API returns JSON + VTK cube, drops straight into ParaView.

Takes <20 min to wire into your optimization script.

Open slot Thu 11 am or Fri 3 pm EST?

Best,
Adam"""
    },
    
    "touch_3": {
        "subject": "Should I close your beta slot?",
        "body": """Hi {first_name},

Beta capped at 10 accounts – 6 taken.

If surrogate-as-a-service isn't on your 2026 roadmap, feel free to ignore.

Otherwise grab a key here: https://surrapi.io/signup

Cheers,
Adam"""
    },
    
    "touch_4": {
        "subject": "Case study: 90% CFD time reduction",
        "body": """Hi {first_name},

A thermal design team just shipped their new housing design.

They ran 200+ Reynolds sweeps in a single afternoon using our API.
Total cost: $50. Previous method: 3 weeks + $4k in cloud HPC.

Happy to walk through how they did it. 20 min?

— Adam

P.S. Their lead engineer wrote a testimonial, can share if helpful."""
    },
    
    "touch_5": {
        "subject": "Closing the loop",
        "body": """Hi {first_name},

I'll assume surrogate CFD isn't a priority right now.

I'll stop following up, but if you ever need instant flow predictions without the queue, I'm at adam@surrapi.io.

Best of luck with your {industry} projects,
Adam"""
    }
}

# Touch schedule (days after previous touch)
TOUCH_SCHEDULE = [0, 3, 7, 14, 21]

# =============================================================================
# Email Sending
# =============================================================================

def send_gmail(to: str, subject: str, body: str, dry_run: bool = False) -> bool:
    """Send email via Gmail SMTP"""
    if dry_run:
        logging.info(f"[DRY RUN] Would send to {to}: {subject}")
        return True
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logging.info(f"✓ Sent to {to}")
        return True
        
    except Exception as e:
        logging.error(f"✗ Failed to send to {to}: {e}")
        return False


def send_sendgrid(to: str, subject: str, body: str, dry_run: bool = False) -> bool:
    """Send email via SendGrid API"""
    if dry_run:
        logging.info(f"[DRY RUN] Would send to {to}: {subject}")
        return True
    
    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail
        
        sg = sendgrid.SendGridAPIClient(api_key=EMAIL_PASSWORD)
        message = Mail(
            from_email=EMAIL_FROM,
            to_emails=to,
            subject=subject,
            plain_text_content=body
        )
        response = sg.send(message)
        
        if response.status_code in [200, 201, 202]:
            logging.info(f"✓ Sent to {to}")
            return True
        else:
            logging.error(f"✗ SendGrid error: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"✗ Failed to send to {to}: {e}")
        return False


def send_email(to: str, subject: str, body: str, dry_run: bool = False) -> bool:
    """Send email using configured provider"""
    if EMAIL_PROVIDER == "sendgrid":
        return send_sendgrid(to, subject, body, dry_run)
    else:
        return send_gmail(to, subject, body, dry_run)


# =============================================================================
# Campaign Management
# =============================================================================

class OutreachCampaign:
    """Manage multi-touch email campaign"""
    
    def __init__(self, leads_csv: str, state_file: str = "outreach/campaign_state.json"):
        self.leads_csv = leads_csv
        self.state_file = Path(state_file)
        self.leads = self._load_leads()
        self.state = self._load_state()
    
    def _load_leads(self) -> List[Dict]:
        """Load leads from CSV"""
        leads = []
        with open(self.leads_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                leads.append(row)
        logging.info(f"Loaded {len(leads)} leads from {self.leads_csv}")
        return leads
    
    def _load_state(self) -> Dict:
        """Load campaign state from JSON"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"sent": {}, "last_run": None}
    
    def _save_state(self):
        """Save campaign state"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _personalize(self, template: Dict, lead: Dict) -> tuple:
        """Personalize template for lead"""
        subject = template["subject"]
        body = template["body"].format(
            first_name=lead.get("first_name", "there"),
            tech_stack=lead.get("tech_stack", "your CFD software"),
            company=lead.get("company", "your company"),
            industry=lead.get("industry", "engineering"),
            title=lead.get("title", "")
        )
        return subject, body
    
    def get_next_touch(self, email: str) -> Optional[int]:
        """Determine next touch number for a lead"""
        if email not in self.state["sent"]:
            return 1
        
        touches = self.state["sent"][email]
        last_touch = max(touches.keys(), key=int)
        last_date = datetime.fromisoformat(touches[last_touch])
        
        next_touch = int(last_touch) + 1
        if next_touch > 5:
            return None  # Campaign complete
        
        # Check if enough time has passed
        days_since = (datetime.now() - last_date).days
        days_needed = TOUCH_SCHEDULE[next_touch - 1]
        
        if days_since >= days_needed:
            return next_touch
        
        return None  # Not time yet
    
    def send_touch(self, lead: Dict, touch_num: int, dry_run: bool = False) -> bool:
        """Send a specific touch to a lead"""
        template = TEMPLATES[f"touch_{touch_num}"]
        subject, body = self._personalize(template, lead)
        email = lead["email"]
        
        success = send_email(email, subject, body, dry_run)
        
        if success and not dry_run:
            if email not in self.state["sent"]:
                self.state["sent"][email] = {}
            self.state["sent"][email][str(touch_num)] = datetime.now().isoformat()
            self._save_state()
        
        return success
    
    def run(self, dry_run: bool = False, max_emails: int = MAX_EMAILS_PER_DAY):
        """Run the campaign, sending due emails"""
        logging.info(f"Starting campaign run (dry_run={dry_run}, max={max_emails})")
        
        sent_count = 0
        
        for lead in self.leads:
            if sent_count >= max_emails:
                logging.info(f"Reached daily limit ({max_emails})")
                break
            
            email = lead.get("email", "")
            if not email or email == "N/A":
                continue
            
            next_touch = self.get_next_touch(email)
            if next_touch is None:
                continue
            
            logging.info(f"Sending touch {next_touch} to {lead['first_name']} at {email}")
            
            if self.send_touch(lead, next_touch, dry_run):
                sent_count += 1
                if not dry_run:
                    time.sleep(DELAY_BETWEEN_EMAILS)
        
        self.state["last_run"] = datetime.now().isoformat()
        self._save_state()
        
        logging.info(f"Campaign run complete. Sent {sent_count} emails.")
        return sent_count
    
    def status(self):
        """Print campaign status"""
        total = len(self.leads)
        not_started = sum(1 for l in self.leads if l["email"] not in self.state["sent"])
        in_progress = sum(1 for l in self.leads 
                        if l["email"] in self.state["sent"] 
                        and max(int(t) for t in self.state["sent"][l["email"]].keys()) < 5)
        complete = sum(1 for l in self.leads 
                      if l["email"] in self.state["sent"] 
                      and max(int(t) for t in self.state["sent"][l["email"]].keys()) >= 5)
        
        print(f"""
📊 Campaign Status
==================
Total leads:     {total}
Not started:     {not_started}
In progress:     {in_progress}
Complete:        {complete}
Last run:        {self.state.get('last_run', 'Never')}
        """)
        
        # Touch breakdown
        touch_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for email, touches in self.state["sent"].items():
            max_touch = max(int(t) for t in touches.keys())
            touch_counts[max_touch] += 1
        
        print("Touch Progress:")
        for t, count in touch_counts.items():
            bar = "█" * count
            print(f"  Touch {t}: {count:3d} {bar}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SurrAPI Email Outreach Campaign")
    parser.add_argument("--csv", default="sales/leads/cfd_prospects_50.csv",
                       help="Path to leads CSV file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview emails without sending")
    parser.add_argument("--max", type=int, default=MAX_EMAILS_PER_DAY,
                       help=f"Max emails to send (default: {MAX_EMAILS_PER_DAY})")
    parser.add_argument("--status", action="store_true",
                       help="Show campaign status and exit")
    parser.add_argument("--reset", action="store_true",
                       help="Reset campaign state (start fresh)")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s"
    )
    
    campaign = OutreachCampaign(args.csv)
    
    if args.status:
        campaign.status()
        return
    
    if args.reset:
        campaign.state = {"sent": {}, "last_run": None}
        campaign._save_state()
        logging.info("Campaign state reset")
        return
    
    if not EMAIL_PASSWORD and not args.dry_run:
        logging.error("Set SURRAPI_EMAIL_PASSWORD environment variable")
        logging.info("For Gmail: Use an App Password from security settings")
        logging.info("For SendGrid: Use your API key")
        sys.exit(1)
    
    campaign.run(dry_run=args.dry_run, max_emails=args.max)


if __name__ == "__main__":
    main()
