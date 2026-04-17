# Terraform — carmen-analysis serving infra

Provisions the public web entry point for the CARMEN Analysis Streamlit app:

```
Internet
   │ HTTPS (443) — GCP-managed cert for carmen-analysis.broadinstitute.org
   ▼
Global External Application LB  (static IP: carmen-analysis-ip)
   │
   ▼
IAP (broadinstitute.org members only)
   │
   ▼
Serverless NEG  ──▶  Cloud Run service (carmen-analysis)
                       └─ container image from Artifact Registry
                          us-central1-docker.pkg.dev/sabeti-adapt/carmen-analysis/...
```

Plus an HTTP (80) forwarding rule that 301-redirects to HTTPS.

## Prerequisites

1. `carmen-analysis-ip` (global external static IP) already exists in
   `sabeti-adapt`. Created out-of-band so the BITS DNS A-record ticket
   could be filed in parallel.
2. The DNS A record `carmen-analysis.broadinstitute.org → <static-ip>` is in
   place. The GCP-managed cert will not finish provisioning until DNS
   resolves to the LB's IP.
3. The container image has been built+pushed by `.github/workflows/docker.yml`.
4. The Cloud OAuth consent screen ("brand") needs a support email — pass it
   via `iap_support_email`. Use a group address you control.

## Apply

```bash
cd terraform
terraform init
terraform plan -var "iap_support_email=carmen-eng@broadinstitute.org" \
               -var 'iap_members=["group:sabeti-lab@broadinstitute.org"]'
terraform apply -var "iap_support_email=carmen-eng@broadinstitute.org" \
                -var 'iap_members=["group:sabeti-lab@broadinstitute.org"]'
```

Or write a `terraform.tfvars`:

```hcl
iap_support_email = "carmen-eng@broadinstitute.org"
iap_members = [
  "group:sabeti-lab@broadinstitute.org",
]
```

After apply, certificate provisioning is asynchronous. Watch it with:

```bash
gcloud compute ssl-certificates describe carmen-analysis-cert \
  --global --project sabeti-adapt --format='value(managed.status)'
```

Status will move from `PROVISIONING` → `ACTIVE` once DNS is live (typically
10–60 minutes).

## Notes

- Cloud Run ingress is set to `INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER`, so
  the service is unreachable except via the LB+IAP front door.
- Only the IAP service principal has `roles/run.invoker`; end users hit IAP,
  IAP forwards to Cloud Run, and IAP enforces the `iap_members` allow-list.
- `enable_cdn = false` — these are diagnostic outputs, not cacheable assets.
- This stack mirrors the pattern from `sabeti-librechat-deployment` but uses
  `google_compute_managed_ssl_certificate` (free, GCP-managed, auto-renewing)
  rather than a BITS-issued cert.
