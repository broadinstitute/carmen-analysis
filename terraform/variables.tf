variable "project_id" {
  description = "GCP project that owns the Cloud Run service and the domain mapping."
  type        = string
  default     = "sabeti-adapt"
}

variable "region" {
  description = "Region for the Cloud Run service and the (regional) artifact registry."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name and the user-visible app slug."
  type        = string
  default     = "carmen-analysis"
}

variable "image" {
  description = "Fully-qualified container image to deploy. Override per-environment in tfvars."
  type        = string
  # The CI workflow pushes a :sha tag; for steady-state deploys we point at :latest.
  default = "us-central1-docker.pkg.dev/sabeti-adapt/carmen-analysis/carmen-analysis:latest"
}

variable "domain" {
  description = "FQDN to map to the Cloud Run service. Cloud Run provisions and renews the managed cert automatically. The BITS DNS record for this name must point at one of Google's frontend IPs (CNAME to ghs.googlehosted.com or A records to the documented anycast pool) — the previous LB static IP does not work."
  type        = string
  default     = "carmen-analysis.broadinstitute.org"
}

variable "cloud_run_min_instances" {
  description = "Minimum Cloud Run instances. 0 = scale to zero."
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Maximum Cloud Run instances. Caps blast radius of misuse."
  type        = number
  default     = 5
}

variable "cloud_run_cpu" {
  description = "CPU per Cloud Run instance."
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Memory per Cloud Run instance."
  type        = string
  default     = "2Gi"
}

variable "cloud_run_timeout_seconds" {
  description = "Per-request timeout. Analyses run 2–5 min in practice; 600s gives headroom."
  type        = number
  default     = 600
}

variable "cloud_run_concurrency" {
  description = "Max concurrent requests per container. Must be >1: Streamlit's persistent websocket plus file-upload XHRs are separate Cloud Run requests, and routing the upload to a fresh sessionless instance returns 400. Session affinity pins a browser to one instance so the pipeline still effectively serializes per-user."
  type        = number
  default     = 80
}

variable "gar_repository_id" {
  description = "Artifact Registry repository ID for the carmen-analysis Docker image."
  type        = string
  default     = "carmen-analysis"
}
