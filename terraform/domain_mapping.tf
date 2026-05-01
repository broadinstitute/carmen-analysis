###############################################################################
# Cloud Run native domain mapping for carmen-analysis.broadinstitute.org.       #
#                                                                               #
# Replaces the LB+cert stack: Cloud Run terminates TLS itself for the mapped    #
# domain, provisions and renews a Google-managed cert under the hood, and       #
# routes traffic directly to the service. No serverless NEG, no backend         #
# service, no URL maps, no forwarding rules.                                    #
#                                                                               #
# DNS prerequisite: carmen-analysis.broadinstitute.org must resolve to Google's  #
# anycast pool via A/AAAA records (managed in dns.tf). CNAME is not possible    #
# because the name is a Cloud DNS zone apex. BITS NS-delegates the subdomain    #
# to our Cloud DNS zone; we own all records under it.                           #
#                                                                               #
# Domain verification prerequisite: carmen-analysis.broadinstitute.org must be  #
# verified in the GCP project. TXT record is in dns.tf; verify with:            #
#   gcloud domains verify carmen-analysis.broadinstitute.org                    #
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
