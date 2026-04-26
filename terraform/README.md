# Terraform — carmen-analysis serving infra

Provisions the public web entry point for the CARMEN Analysis Streamlit app:

```
Internet
   │ HTTPS (443) — Google-managed cert (auto-provisioned by Cloud Run)
   ▼
Cloud Run domain mapping  (carmen-analysis.broadinstitute.org)
   │
   ▼
Cloud Run service (carmen-analysis, ingress=ALL, allUsers run.invoker)
   └─ container image from Artifact Registry
      us-central1-docker.pkg.dev/sabeti-adapt/carmen-analysis/...
```

The site is **public** — no authentication. The data passing through the
pipeline is on the user's laptop and nothing is retained server-side, so the
auth tier wasn't load-bearing. The earlier IAP design was abandoned because
Google deprecated the IAP OAuth Admin API in July 2025.

The previous LB+cert front door is gone; Cloud Run's native domain mapping
terminates TLS itself, provisions and renews the managed cert under the hood,
and routes traffic directly to the service. No load balancer, no serverless
NEG, no static IP, no managed-cert resource.

## Prerequisites

1. **Domain verification.** `broadinstitute.org` must be verified for the
   `sabeti-adapt` GCP project (or for an org/Workspace it belongs to). Check:
   ```bash
   gcloud domains list-user-verified
   ```
   If absent, run the verification flow at
   https://console.cloud.google.com/run/domains?project=sabeti-adapt — Google
   gives you a TXT record to publish via BITS.
2. **DNS.** `carmen-analysis.broadinstitute.org` must resolve to one of
   Google's frontend IPs. Either:
   - `CNAME ghs.googlehosted.com.` (preferred for subdomains), or
   - A records to `216.239.32.21`, `216.239.34.21`, `216.239.36.21`,
     `216.239.38.21` plus the corresponding AAAA records.
   The previous LB static IP (`carmen-analysis-ip` / `35.241.20.121`) does
   **not** work — Cloud Run domain mappings only accept the shared frontend
   pool. The cert will not provision until DNS resolves.
3. The container image has been built+pushed by `.github/workflows/docker.yml`.

## Apply

`terraform.tfvars` can be empty — all variables have sensible defaults for the
standard `sabeti-adapt` deployment. Override only what you need.

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

After apply, certificate provisioning is asynchronous. Watch it with:

```bash
gcloud beta run domain-mappings describe \
  --domain=carmen-analysis.broadinstitute.org \
  --region=us-central1 \
  --project=sabeti-adapt \
  --format='value(status.conditions)'
```

The mapping reports `Ready: False` until DNS resolves and the cert is in
place; once both are good, it flips to `Ready: True` (typically 10–60 min
after DNS lands).

## Staging service

`staging.tf` provisions a second Cloud Run service, `carmen-analysis-staging`,
on the same posture as production (public, no auth) but with no domain
mapping — it's reached via its `*.run.app` URL only:

- `ingress = INGRESS_TRAFFIC_ALL` (reachable directly at `*.run.app`)
- `allUsers` granted `roles/run.invoker` (no auth)
- `lifecycle.ignore_changes = [template]` — Terraform owns the service shell
  and IAM only; CI owns the live revisions.

Every push to a non-tag ref triggers `.github/workflows/docker.yml`'s
`staging-deploy` job, which sanitizes the branch name into a Cloud Run
revision tag and runs `gcloud run deploy --tag <branch> --no-traffic`. The
URL is `https://<branch>---carmen-analysis-staging-<hash>.us-central1.run.app`
and is surfaced in the workflow run's summary tab.

This service is **deliberately public and unauth'd** — never deploy anything
sensitive to it. Treat it as a throwaway sandbox. PRs from forks do not get a
staging deploy (their workflow runs don't have access to the WIF secrets).

## Notes

- The production service's `*.run.app` URL is also publicly reachable
  (ingress=ALL), but the FQDN is the canonical entry point.
- The previous BITS-managed DNS A record to `carmen-analysis-ip` /
  `35.241.20.121` should be replaced with a CNAME (or A records to the
  Google frontend pool) before this terraform will fully come up. The
  static IP can be released after migration.
