#!/bin/bash
# =============================================================================
# Deploy SurrAPI to Railway
# =============================================================================
#
# Prerequisites:
#   1. Install Railway CLI: npm install -g @railway/cli
#   2. Login: railway login
#   3. Create project: railway init
#
# Usage:
#   ./deploy/railway_deploy.sh
# =============================================================================

set -e

echo "🚂 Deploying SurrAPI to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install with: npm install -g @railway/cli"
    exit 1
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in. Run: railway login"
    exit 1
fi

# Set environment variables
echo "📦 Setting environment variables..."
railway variables set SURRAPI_PORT=8000
railway variables set SURRAPI_DEVICE=auto
railway variables set SURRAPI_LOG_LEVEL=INFO
railway variables set SURRAPI_CHECKPOINT=app/assets/fno_128.pt

# Deploy
echo "🚀 Deploying..."
railway up --detach

# Get deployment URL
echo ""
echo "✅ Deployment initiated!"
echo ""
echo "View logs:    railway logs"
echo "Open app:     railway open"
echo "Get domain:   railway domain"
echo ""
echo "📋 Next steps:"
echo "   1. Add custom domain: railway domain add api.surrapi.io"
echo "   2. Enable GPU: Upgrade to Pro plan for GPU instances"
echo "   3. Add monitoring: railway add newrelic (or similar)"
