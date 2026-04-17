output "service_url" {
  description = "Public URL of the web app once DNS + cert provision."
  value       = "https://${var.domain}"
}

output "lb_ip_address" {
  description = "Static IP the LB serves traffic on. The BITS DNS A record must point here."
  value       = data.google_compute_global_address.lb_ip.address
}

output "cloud_run_service_name" {
  value = google_cloud_run_v2_service.app.name
}

output "managed_cert_name" {
  description = "Inspect provisioning state with: gcloud compute ssl-certificates describe <name>."
  value       = google_compute_managed_ssl_certificate.app.name
}

output "staging_service_name" {
  description = "Cloud Run staging service. CI deploys per-branch tagged revisions to it."
  value       = google_cloud_run_v2_service.staging.name
}

output "staging_base_url" {
  description = "Base *.run.app URL of the staging service. Per-branch URLs prepend `<branch>---`."
  value       = google_cloud_run_v2_service.staging.uri
}
