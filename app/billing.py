"""
SurrAPI Billing Module
======================

Stripe integration for usage-based billing.
Tracks API calls and charges per prediction.

Environment Variables:
    STRIPE_SECRET_KEY - Stripe secret key (sk_...)
    STRIPE_WEBHOOK_SECRET - Webhook signing secret (whsec_...)
    STRIPE_PRICE_ID - Price ID for pay-as-you-go ($0.25/prediction)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from functools import wraps

import stripe
from fastapi import HTTPException, Header, Depends, Request
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "price_surrapi_prediction")

# Pricing tiers
TIERS = {
    "free": {
        "name": "Starter",
        "price": 0,
        "predictions_per_month": 1000,
        "rate_limit_per_second": 5,
        "max_resolution": 128
    },
    "pro": {
        "name": "Pro",
        "price": 199,
        "predictions_per_month": 50000,
        "rate_limit_per_second": 50,
        "max_resolution": 256
    },
    "enterprise": {
        "name": "Enterprise",
        "price": None,  # Custom
        "predictions_per_month": None,  # Unlimited
        "rate_limit_per_second": 100,
        "max_resolution": 512
    }
}

# Initialize Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    logging.warning("STRIPE_SECRET_KEY not set - billing disabled")

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class APIKey:
    """API key with usage tracking"""
    key: str
    user_id: str
    tier: str = "free"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Usage tracking
    predictions_this_month: int = 0
    predictions_total: int = 0
    month_start: datetime = field(default_factory=lambda: datetime.now().replace(day=1))
    
    # Stripe
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    
    def reset_monthly(self):
        """Reset monthly counter if new month"""
        now = datetime.now()
        if now.month != self.month_start.month or now.year != self.month_start.year:
            self.predictions_this_month = 0
            self.month_start = now.replace(day=1)
    
    def can_predict(self) -> bool:
        """Check if this key can make a prediction"""
        self.reset_monthly()
        limit = TIERS[self.tier]["predictions_per_month"]
        if limit is None:  # Unlimited
            return True
        return self.predictions_this_month < limit
    
    def record_prediction(self):
        """Record a prediction for billing"""
        self.predictions_this_month += 1
        self.predictions_total += 1


class CreateKeyRequest(BaseModel):
    email: str
    tier: str = "free"


class CreateKeyResponse(BaseModel):
    api_key: str
    tier: str
    predictions_per_month: int
    rate_limit_per_second: int


class UsageResponse(BaseModel):
    predictions_this_month: int
    predictions_total: int
    limit_per_month: Optional[int]
    tier: str
    reset_date: str


class WebhookEvent(BaseModel):
    type: str
    data: dict


# =============================================================================
# API Key Store (In-memory for demo, use Redis/DB in production)
# =============================================================================

_api_keys: Dict[str, APIKey] = {}


def generate_api_key() -> str:
    """Generate a secure API key"""
    import secrets
    return f"sk_{secrets.token_urlsafe(32)}"


def create_api_key(email: str, tier: str = "free") -> APIKey:
    """Create a new API key for a user"""
    key = generate_api_key()
    api_key = APIKey(
        key=key,
        user_id=email,
        tier=tier
    )
    _api_keys[key] = api_key
    logging.info(f"Created API key for {email}: {key[:12]}...")
    return api_key


def get_api_key(key: str) -> Optional[APIKey]:
    """Retrieve an API key"""
    return _api_keys.get(key)


def validate_api_key(key: str) -> APIKey:
    """Validate API key and check quota"""
    api_key = get_api_key(key)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not api_key.can_predict():
        raise HTTPException(
            status_code=429,
            detail=f"Monthly quota exceeded. Upgrade to Pro for more predictions."
        )
    
    return api_key


# =============================================================================
# Authentication Dependency
# =============================================================================

async def get_current_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
) -> Optional[APIKey]:
    """
    Extract and validate API key from headers.
    Supports both Authorization: Bearer and X-API-Key headers.
    """
    key = None
    
    if authorization:
        # Bearer token
        if authorization.startswith("Bearer "):
            key = authorization[7:]
        else:
            key = authorization
    elif x_api_key:
        key = x_api_key
    
    if not key:
        # Allow unauthenticated requests for demo
        return None
    
    return validate_api_key(key)


def require_api_key(func):
    """Decorator to require API key authentication"""
    @wraps(func)
    async def wrapper(*args, api_key: APIKey = Depends(get_current_key), **kwargs):
        if api_key is None:
            raise HTTPException(
                status_code=401,
                detail="API key required. Get one at https://surrapi.io/signup"
            )
        return await func(*args, api_key=api_key, **kwargs)
    return wrapper


# =============================================================================
# Stripe Integration
# =============================================================================

async def create_stripe_customer(email: str) -> str:
    """Create a Stripe customer"""
    if not STRIPE_SECRET_KEY:
        return "demo_customer"
    
    customer = stripe.Customer.create(
        email=email,
        metadata={"source": "surrapi"}
    )
    return customer.id


async def create_checkout_session(
    customer_id: str,
    tier: str,
    success_url: str = "https://surrapi.io/success",
    cancel_url: str = "https://surrapi.io/pricing"
) -> str:
    """Create a Stripe Checkout session for subscription"""
    if not STRIPE_SECRET_KEY:
        return "https://demo.stripe.com/checkout"
    
    # Get price ID for tier
    price_ids = {
        "pro": os.getenv("STRIPE_PRO_PRICE_ID", "price_pro_monthly"),
        "enterprise": os.getenv("STRIPE_ENTERPRISE_PRICE_ID", "price_enterprise")
    }
    
    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{
            "price": price_ids.get(tier, price_ids["pro"]),
            "quantity": 1
        }],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url
    )
    
    return session.url


async def create_usage_record(
    subscription_item_id: str,
    quantity: int = 1
) -> None:
    """Report usage to Stripe for metered billing"""
    if not STRIPE_SECRET_KEY:
        return
    
    stripe.SubscriptionItem.create_usage_record(
        subscription_item_id,
        quantity=quantity,
        timestamp=int(datetime.now().timestamp()),
        action="increment"
    )


async def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """Handle Stripe webhook events"""
    if not STRIPE_SECRET_KEY or not STRIPE_WEBHOOK_SECRET:
        return {"received": True}
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle specific events
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session["customer"]
        subscription_id = session["subscription"]
        
        # Update API key with subscription
        for key in _api_keys.values():
            if key.stripe_customer_id == customer_id:
                key.stripe_subscription_id = subscription_id
                key.tier = "pro"  # Upgrade tier
                logging.info(f"Upgraded {key.user_id} to Pro")
                break
    
    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        subscription_id = subscription["id"]
        
        # Downgrade to free
        for key in _api_keys.values():
            if key.stripe_subscription_id == subscription_id:
                key.tier = "free"
                key.stripe_subscription_id = None
                logging.info(f"Downgraded {key.user_id} to Free")
                break
    
    elif event["type"] == "invoice.payment_failed":
        invoice = event["data"]["object"]
        customer_id = invoice["customer"]
        
        # Handle failed payment - could suspend access
        logging.warning(f"Payment failed for customer {customer_id}")
    
    return {"received": True}


# =============================================================================
# FastAPI Routes (to be added to main.py)
# =============================================================================

def register_billing_routes(app):
    """Register billing routes with FastAPI app"""
    
    @app.post("/api/keys", response_model=CreateKeyResponse)
    async def create_key(request: CreateKeyRequest):
        """Create a new API key"""
        # Create Stripe customer if billing enabled
        customer_id = None
        if STRIPE_SECRET_KEY:
            try:
                customer_id = await create_stripe_customer(request.email)
            except stripe.error.StripeError as e:
                logging.error(f"Stripe error: {e}")
                # For free tier, we might allow creation without Stripe?
                # But let's fail safe for now
                raise HTTPException(
                    status_code=503, 
                    detail="Billing service unavailable. Please try again later."
                )
        
        api_key = create_api_key(request.email, request.tier)
        if customer_id:
            api_key.stripe_customer_id = customer_id
        
        tier_info = TIERS[api_key.tier]
        return CreateKeyResponse(
            api_key=api_key.key,
            tier=api_key.tier,
            predictions_per_month=tier_info["predictions_per_month"] or 999999,
            rate_limit_per_second=tier_info["rate_limit_per_second"]
        )
    
    @app.get("/api/usage", response_model=UsageResponse)
    async def get_usage(api_key: APIKey = Depends(get_current_key)):
        """Get current usage for API key"""
        if api_key is None:
            raise HTTPException(status_code=401, detail="API key required")
        
        tier_info = TIERS[api_key.tier]
        next_month = (api_key.month_start + timedelta(days=32)).replace(day=1)
        
        return UsageResponse(
            predictions_this_month=api_key.predictions_this_month,
            predictions_total=api_key.predictions_total,
            limit_per_month=tier_info["predictions_per_month"],
            tier=api_key.tier,
            reset_date=next_month.isoformat()
        )
    
    @app.post("/api/upgrade")
    async def upgrade_subscription(
        tier: str = "pro",
        api_key: APIKey = Depends(get_current_key)
    ):
        """Get Stripe checkout URL for upgrade"""
        if api_key is None:
            raise HTTPException(status_code=401, detail="API key required")
        
        if not api_key.stripe_customer_id:
            customer_id = await create_stripe_customer(api_key.user_id)
            api_key.stripe_customer_id = customer_id
        
        checkout_url = await create_checkout_session(
            api_key.stripe_customer_id,
            tier
        )
        
        return {"checkout_url": checkout_url}
    
    @app.post("/webhooks/stripe")
    async def stripe_webhook(request: Request):
        """Handle Stripe webhook events"""
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        return await handle_webhook(payload, sig_header)
    
    logging.info("Billing routes registered")


# =============================================================================
# Middleware for Usage Tracking
# =============================================================================

class BillingMiddleware:
    """
    Middleware to track API usage and record to Stripe.
    Add to FastAPI app with: app.add_middleware(BillingMiddleware)
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Track prediction endpoints
        path = scope.get("path", "")
        if "/predict" in path:
            # Extract API key from headers
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            x_api_key = headers.get(b"x-api-key", b"").decode()
            
            key = None
            if auth.startswith("Bearer "):
                key = auth[7:]
            elif x_api_key:
                key = x_api_key
            
            if key:
                api_key = get_api_key(key)
                if api_key:
                    api_key.record_prediction()
                    
                    # Report to Stripe for metered billing
                    if api_key.stripe_subscription_id and STRIPE_SECRET_KEY:
                        # Would need subscription item ID in production
                        pass
        
        await self.app(scope, receive, send)
