# Email Outreach Setup Guide

## Quick Start

### 1. Gmail Setup (Easiest)

1. Enable 2-Factor Authentication on your Google account
2. Go to: https://myaccount.google.com/apppasswords
3. Generate a new App Password for "Mail"
4. Set environment variables:

```bash
export SURRAPI_EMAIL_FROM="your-email@gmail.com"
export SURRAPI_EMAIL_PASSWORD="xxxx-xxxx-xxxx-xxxx"  # 16-char app password
export SURRAPI_EMAIL_PROVIDER="gmail"
```

### 2. SendGrid Setup (More Professional)

1. Sign up at https://sendgrid.com (free tier: 100 emails/day)
2. Create API key: Settings → API Keys → Create
3. Verify your sending domain for better deliverability
4. Set environment variables:

```bash
export SURRAPI_EMAIL_FROM="team@surrapi.io"
export SURRAPI_EMAIL_PASSWORD="SG.xxxxxxxxxxxx"  # SendGrid API key
export SURRAPI_EMAIL_PROVIDER="sendgrid"

pip install sendgrid
```

## Running the Campaign

### Preview Mode (Recommended First)

```bash
cd /Users/adamsussman/Desktop/Active_Projects/surrapi-demo
source venv/bin/activate

# See all emails that would be sent
python outreach/send_campaign.py --dry-run

# Check campaign status
python outreach/send_campaign.py --status
```

### Send Emails

```bash
# Send first batch (default: 30/day)
python outreach/send_campaign.py

# Send more
python outreach/send_campaign.py --max 50

# Run daily via cron
0 9 * * * cd /path/to/surrapi-demo && python outreach/send_campaign.py >> logs/outreach.log 2>&1
```

## Email Sequence

| Touch | Subject | Timing |
|-------|---------|--------|
| 1 | "300 ms CFD instead of 6 h queue?" | Day 0 |
| 2 | "Re: 300 ms CFD – accuracy data attached" | Day 3 |
| 3 | "Should I close your beta slot?" | Day 7 |
| 4 | "Case study: 90% CFD time reduction" | Day 14 |
| 5 | "Closing the loop" | Day 21 |

## Tracking Replies

For now, manually check your inbox for replies. When someone replies:

1. Mark them in the CSV (add a "replied" column)
2. Remove from future touches by resetting their state
3. Schedule a call via Calendly

## Expected Results

Based on B2B cold email benchmarks:
- Open rate: 15-25%
- Reply rate: 2-5%
- Meeting rate: 0.5-1%

With 50 leads:
- ~1-3 positive replies expected
- First pilot customer typically from touches 1-3
