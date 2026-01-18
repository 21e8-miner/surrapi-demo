# Stripe Billing Setup Guide

## Overview

SurrAPI uses Stripe for:
1. **API Key Generation** - Free tier signup
2. **Subscription Billing** - Pro tier ($199/month)
3. **Usage-Based Billing** - Pay-per-prediction overage

## Quick Setup

### 1. Create Stripe Account

1. Sign up at https://stripe.com
2. Get API keys from Developers → API Keys
3. Set environment variables:

```bash
# Test keys (for development)
export STRIPE_SECRET_KEY="sk_test_..."
export STRIPE_PUBLISHABLE_KEY="pk_test_..."

# Production keys (for live billing)
export STRIPE_SECRET_KEY="sk_live_..."
export STRIPE_PUBLISHABLE_KEY="pk_live_..."
```

### 2. Create Products/Prices

In Stripe Dashboard → Products:

**Product 1: SurrAPI Pro**
- Name: SurrAPI Pro Monthly
- Price: $199/month (recurring)
- Price ID: `price_pro_monthly`

**Product 2: Prediction (Metered)**
- Name: SurrAPI Prediction
- Price: $0.25/unit (metered)
- Usage type: Metered
- Price ID: `price_prediction`

```bash
export STRIPE_PRO_PRICE_ID="price_pro_monthly"
export STRIPE_PREDICTION_PRICE_ID="price_prediction"
```

### 3. Configure Webhooks

1. Go to Developers → Webhooks
2. Add endpoint: `https://api.surrapi.io/webhooks/stripe`
3. Select events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`

4. Copy webhook signing secret:

```bash
export STRIPE_WEBHOOK_SECRET="whsec_..."
```

### 4. Integrate with SurrAPI

The billing module is already integrated. Just add to `main.py`:

```python
from app.billing import register_billing_routes

# After app creation
register_billing_routes(app)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/keys` | POST | Create new API key (requires email) |
| `/api/usage` | GET | Get current usage stats |
| `/api/upgrade` | POST | Get Stripe checkout URL for Pro |
| `/webhooks/stripe` | POST | Handle Stripe webhook events |

## Testing

### Create Test API Key

```bash
curl -X POST "https://api.surrapi.io/api/keys" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

Response:
```json
{
  "api_key": "sk_abc123...",
  "tier": "free",
  "predictions_per_month": 1000,
  "rate_limit_per_second": 5
}
```

### Check Usage

```bash
curl "https://api.surrapi.io/api/usage" \
  -H "Authorization: Bearer sk_abc123..."
```

### Test Stripe Checkout

```bash
# Use Stripe test card: 4242 4242 4242 4242
curl -X POST "https://api.surrapi.io/api/upgrade?tier=pro" \
  -H "Authorization: Bearer sk_abc123..."
```

## Pricing Tiers

| Tier | Monthly | Predictions | Rate Limit | Max Resolution |
|------|---------|-------------|------------|----------------|
| Free | $0 | 1,000 | 5/sec | 128×128 |
| Pro | $199 | 50,000 | 50/sec | 256×256 |
| Enterprise | Custom | Unlimited | 100/sec | 512×512 |

## Production Checklist

- [ ] Switch to live API keys
- [ ] Configure production webhook URL
- [ ] Set up Stripe Tax (if applicable)
- [ ] Enable fraud protection (Radar)
- [ ] Configure receipt emails
- [ ] Set up usage alerts
- [ ] Implement Redis for API key storage (replace in-memory)
