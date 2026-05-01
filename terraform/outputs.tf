output "service_url" {
  description = "Public URL of the web app once DNS + Cloud Run domain mapping cert provision."
  value       = "https://${var.domain}"
}

output "cloud_run_service_name" {
  value = google_cloud_run_v2_service.app.name
}

output "cloud_run_default_url" {
  description = "Direct *.run.app URL of the production service (always reachable since ingress=ALL)."
  value       = google_cloud_run_v2_service.app.uri
}

output "staging_service_name" {
  description = "Cloud Run staging service. CI deploys per-branch tagged revisions to it."
  value       = google_cloud_run_v2_service.staging.name
}

output "staging_base_url" {
  description = "Base *.run.app URL of the staging service. Per-branch URLs prepend `<branch>---`."
  value       = google_cloud_run_v2_service.staging.uri
}

output "dns_nameservers" {
  description = "NS servers to give BITS for delegation of the carmen-analysis.broadinstitute.org zone."
  value       = google_dns_managed_zone.carmen_analysis.name_servers
}
