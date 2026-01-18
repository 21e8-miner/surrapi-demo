# SurrAPI Cold Outreach Email Templates

## Overview
These templates are designed for CFD consultants and engineers using Ansys, Siemens, or Dassault tools.
Personalize the bracketed fields before sending.

---

## Touch 1: Initial Outreach (Day 0)

**Subject Line Options:**
- 300 ms CFD instead of 6 h queue?
- What if your next CFD run took 0.3 seconds?
- Kill your core-hour budget (in a good way)

**Body:**

```
Hi {FirstName},

Running {Fluent|STAR-CCM+|PowerFLOW} burns ~$120 and half a day every design loop.

We pre-trained an FNO on 15 TB of validated turbulence data. Now you get velocity/pressure fields in 300 ms through a REST call.

Still costs $0.25 but saves 30-300 core-hours.

Swagger demo: https://demo.surrapi.io
Worth 15 min next week?

Best,
{YourName}
{YourTitle}, SurrAPI
```

---

## Touch 2: Follow-Up with Proof (Day 3)

**Subject:** Re: 300 ms CFD – graph attached

**Body:**

```
{FirstName},

Screenshot shows drag coefficient error <1% vs full solve for the 2D NACA-2412 set.

API returns JSON + VTK cube so you can drop it straight into ParaView.

Takes <20 min to wire into your optimization script.

Open slot Thu 11 am or Fri 3 pm EST?

Br,
{YourName}
```

**Attachment:** accuracy_chart.png (drag error comparison)

---

## Touch 3: Scarcity Close (Day 7)

**Subject:** Should I close your beta slot?

**Body:**

```
Hi {FirstName},

Beta capped at 10 accounts – 6 taken.

If surrogate-as-a-service isn't on your 2026 roadmap, feel free to ignore.

Otherwise grab a key here: https://surrapi.io/signup

Cheers,
{YourName}
```

---

## Touch 4: Value Add (Day 14)

**Subject:** Case study: {SimilarCompany} cut CFD time 90%

**Body:**

```
{FirstName},

{SimilarCompany} just shipped their new wing design.

They ran 200+ Reynolds sweeps in a single afternoon using our API.
Total cost: $50. Previous method: 3 weeks + $4k in cloud HPC.

Happy to walk through how they did it. 20 min?

— {YourName}

P.S. Their CAE lead wrote a testimonial, can share if helpful.
```

---

## Touch 5: Breakup (Day 21)

**Subject:** Closing the loop

**Body:**

```
{FirstName},

I'll assume surrogate CFD isn't a priority right now.

I'll stop following up, but if you ever need instant flow predictions without the queue, I'm at {YourEmail}.

Best of luck with {upcomingProject/product},
{YourName}
```

---

## LinkedIn Connection Request Template

**Note:** Keep under 300 characters

```
Hi {FirstName} – saw your work on {SpecificProject} at {Company}. We built an API that returns CFD results in 300ms instead of hours. Thought you might find it useful for faster design iteration. Would love to connect!
```

---

## Call Script (After Positive Reply)

### Opening (15 sec)
"Thanks for taking the call, {FirstName}. I know you're busy, so I'll keep it tight.

Quick agenda: I'll show you how the API works, you tell me if it fits your workflow, and we'll go from there. Sound good?"

### Discovery (2 min)
- "What CFD tool do you use today?"
- "How many simulations do you run per project?"
- "What's your biggest pain point – cost, time, or expertise?"

### Demo (5 min)
- Show Swagger live
- Send a prediction
- Display result in Plotly/ParaView

### Close (2 min)
"Looks like this could save you {X} hours per project.

Want me to set you up with a free API key so you can try it on your next design?"

### Objection Handling

**"It's not accurate enough for production."**
> "Totally fair. That's why we publish our validation numbers openly. For preliminary design and optimization loops, 1% error is good enough. For final sign-off, you'd still run the full solve – but you'd run it on the right design the first time."

**"We already have an HPC cluster."**
> "Great – this isn't meant to replace that. It's meant to reduce how many times you use it. Exploratory runs in seconds, final validation on HPC."

**"My boss won't approve a new vendor."**
> "Understood. The freemium tier is 1,000 calls/month – you can test it without any PO. Once you prove value, we can talk enterprise."

---

## Follow-Up After Demo

**Subject:** Your SurrAPI API key + next steps

**Body:**

```
{FirstName},

Great chatting today. As promised:

🔑 Your API Key: sk-XXXXXXXXXXXXXXXX
📖 Quickstart: https://docs.surrapi.io/quickstart
💬 Slack invite: https://surrapi.io/slack

Try it on your next Re sweep and let me know how it goes.

I'll check back Thursday to see if you hit any snags.

— {YourName}
```

---

*Templates last updated: January 2026*
