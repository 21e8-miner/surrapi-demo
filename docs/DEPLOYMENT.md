# SurrAPI Deployment Guide

## Quick Deploy to Railway (Recommended)

### Step 1: Login to Railway
```bash
cd /Users/adamsussman/Desktop/Active_Projects/surrapi-demo
railway login
```
Complete authentication in browser, then:

### Step 2: Initialize Project
```bash
railway init
# Select "Empty Project" or create new
```

### Step 3: Deploy
```bash
railway up
```

### Step 4: Get Domain
```bash
railway domain
# Outputs: surrapi-xxx.up.railway.app
```

### Step 5: Set Environment Variables
```bash
railway variables set STRIPE_SECRET_KEY="sk_live_xxx"
railway variables set STRIPE_WEBHOOK_SECRET="whsec_xxx"
railway variables set SURRAPI_LOG_LEVEL="INFO"
```

### Step 6: Add Custom Domain (Optional)
```bash
railway domain add api.surrapi.io
```

DNS Configuration:
```
Type: CNAME
Name: api
Value: surrapi-xxx.up.railway.app
TTL: 300
```

---

## Alternative: Deploy to Render

### One-Click Deploy
1. Go to https://render.com/deploy
2. Connect GitHub repo
3. Select `render.yaml` blueprint
4. Click Deploy

---

## Alternative: Deploy to Fly.io

### Step 1: Install flyctl
```bash
brew install flyctl
```

### Step 2: Login and Deploy
```bash
fly auth login
fly launch --name surrapi
```

---

## Post-Deployment Checklist

- [ ] Verify `/health` endpoint returns `{"status": "ok"}`
- [ ] Test prediction endpoint
- [ ] Configure Stripe webhook URL to production domain
- [ ] Update CORS origins to production domain only
- [ ] Send first email batch: `python outreach/send_campaign.py --max 10`

---

## Current Status

✅ **Git Repository**: Initialized at `/Users/adamsussman/Desktop/Active_Projects/surrapi-demo`
✅ **Code Hardened**: Rate limiting, security headers, metrics, error handling
✅ **Railway CLI**: Installed (v4.25.2)
⏳ **Deployment**: Awaiting Railway authentication

## Local Testing

The server is currently running at http://localhost:8080
- Landing page: http://localhost:8080
- API Docs: http://localhost:8080/docs
- Metrics: http://localhost:8080/metrics
- Health: http://localhost:8080/health
