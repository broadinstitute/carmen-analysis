###############################################################################
# Cloud Run native domain mapping for carmen-analysis.sabeti.broadinstitute.org #
#                                                                               #
# Replaces the LB+cert stack: Cloud Run terminates TLS itself for the mapped    #
# domain, provisions and renews a Google-managed cert under the hood, and       #
# routes traffic directly to the service. No serverless NEG, no backend         #
# service, no URL maps, no forwarding rules.                                    #
#                                                                               #
# DNS prerequisite: carmen-analysis.sabeti.broadinstitute.org must resolve via  #
# CNAME to ghs.googlehosted.com. DNS is managed in the sabeti.broadinstitute.org#
# Cloud DNS zone (different GCP project).                                       #
#                                                                               #
# Domain verification prerequisite: carmen-analysis.sabeti.broadinstitute.org   #
# (or its parent sabeti.broadinstitute.org) must be verified in THIS project    #
# (sabeti-adapt), not just the project hosting the DNS zone. Verify with:       #
#   gcloud domains verify sabeti.broadinstitute.org --project=sabeti-adapt      #
###############################################################################

resource "google_cloud_run_domain_mapping" "app" {
  name     = var.domain
  location = var.region
  project  = var.project_id

  metadata {
    namespace = var.project_id
  }

  spec {
    route_name = google_cloud_run_v2_service.app.name
  }
}

output "domain_mapping_dns_records" {
  description = "DNS records to publish for the mapped domain. Hand these to BITS."
  value       = google_cloud_run_domain_mapping.app.status[0].resource_records
}
