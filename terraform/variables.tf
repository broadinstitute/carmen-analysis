variable "project_id" {
  description = "GCP project that owns the Cloud Run service, LB, and managed cert."
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
  default     = "us-central1-docker.pkg.dev/sabeti-adapt/carmen-analysis/carmen-analysis:latest"
}

variable "domain" {
  description = "FQDN that the GCP-managed cert is provisioned for. The BITS-managed DNS A record must point at the static IP referenced below."
  type        = string
  default     = "carmen-analysis.broadinstitute.org"
}

variable "static_ip_name" {
  description = "Name of the global external static IP already created in this project."
  type        = string
  default     = "carmen-analysis-ip"
}

variable "iap_members" {
  description = "Principals (user:foo@broadinstitute.org, group:bar@broadinstitute.org) granted IAP access to the web app."
  type        = list(string)
  default     = []
}

variable "iap_support_email" {
  description = "Support email shown on the IAP consent screen. Must be a Group or the deploying user."
  type        = string
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
  description = "Max concurrent requests per container. The pipeline is CPU-bound; one upload at a time."
  type        = number
  default     = 1
}

variable "gar_repository_id" {
  description = "Artifact Registry repository ID for the carmen-analysis Docker image."
  type        = string
  default     = "carmen-analysis"
}
