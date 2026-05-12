# Terraform — carmen-analysis serving infra

Provisions the public web entry point for the CARMEN Analysis Streamlit app:

```
Internet
   │ HTTPS (443) — Google-managed cert (auto-provisioned by Cloud Run)
   ▼
Cloud Run domain mapping  (carmen-analysis.sabeti.broadinstitute.org)
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

1. **Domain verification.** `sabeti.broadinstitute.org` (or the specific subdomain)
   must be verified in the `sabeti-adapt` GCP project — even if the DNS zone is
   managed in a different project. Verifying the parent domain covers all subdomains:
   ```bash
   gcloud domains verify sabeti.broadinstitute.org --project=sabeti-adapt
   gcloud domains list-user-verified --project=sabeti-adapt
   ```
2. **DNS.** `carmen-analysis.sabeti.broadinstitute.org` is managed in the
   `sabeti.broadinstitute.org` Cloud DNS zone (different GCP project). Create a
   CNAME record pointing to `ghs.googlehosted.com`:
   ```
   Name: carmen-analysis.sabeti.broadinstitute.org
   Type: CNAME
   TTL: 300
   Data: ghs.googlehosted.com.
   ```
   The cert will not provision until DNS resolves correctly.
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
  --domain=carmen-analysis.sabeti.broadinstitute.org \
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
- DNS for the FQDN is managed in the `sabeti.broadinstitute.org` Cloud DNS
  zone (different GCP project). A single CNAME record pointing to
  `ghs.googlehosted.com` is all that's needed.
