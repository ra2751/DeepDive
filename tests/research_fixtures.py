"""Synthetic, non-confidential fixtures for the Phase 2 research-graph tests."""

from agents.research_common import evidence_id

CONTRACT = {"text": "The master supply agreement between Northwind Foods and Acme Metals runs until 31 December 2027.",
            "source": "contract.pdf", "page": 1}
SCORECARD = {"text": "Acme Metals delivered 98 percent of purchase orders on time during 2025.",
             "source": "supplier_scorecard.pdf", "page": 4}
AUDIT = {"text": "Acme Metals does not hold ISO 27001 certification; the certificate lapsed in March 2024.",
         "source": "audit_report.pdf", "page": 7}
PRICING = {"text": "Acme Metals raised steel coil prices by 6 percent in January 2026.",
           "source": "pricing_update.pdf", "page": 2}
PROCUREMENT_DOCS = [CONTRACT, SCORECARD, AUDIT, PRICING]

TRIAL_A = {"text": "In the phase 3 trial, Drug X reduced HbA1c by 1.2 percent versus placebo at 26 weeks.",
           "source": "trial_a.pdf", "page": 3}
TRIAL_B = {"text": "A second independent phase 3 trial found Drug X reduced HbA1c by 1.2 percent at 26 weeks.",
           "source": "trial_b.pdf", "page": 5}


def eid(doc):
    return evidence_id(doc["source"], doc["page"], doc["text"])


def judgment(claim_id, doc, stance, quote, reasoning="test"):
    return {"claim_id": claim_id, "evidence_id": eid(doc), "stance": stance, "quote": quote, "reasoning": reasoning}
