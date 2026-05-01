###############################################################################
# Cloud DNS zone delegated from BITS for carmen-analysis.broadinstitute.org.   #
#                                                                               #
# BITS NS-delegates this subdomain to the zone below, giving us full control   #
# over its records without further BITS involvement. The CNAME record is        #
# pre-populated so there is no resolution gap when delegation cuts over.        #
###############################################################################

resource "google_dns_managed_zone" "carmen_analysis" {
  name        = "carmen-analysis-broadinstitute-org"
  dns_name    = "${var.domain}."
  description = "Delegated zone for ${var.domain}"
  project     = var.project_id

  depends_on = [google_project_service.services]
}

# Points the FQDN at Google's shared frontend pool for Cloud Run domain mapping.
# Must be A/AAAA records (not CNAME) because this is the zone apex.
resource "google_dns_record_set" "a" {
  name         = "${var.domain}."
  managed_zone = google_dns_managed_zone.carmen_analysis.name
  type         = "A"
  ttl          = 300
  rrdatas = [
    "216.239.32.21",
    "216.239.34.21",
    "216.239.36.21",
    "216.239.38.21",
  ]
  project = var.project_id
}

resource "google_dns_record_set" "aaaa" {
  name         = "${var.domain}."
  managed_zone = google_dns_managed_zone.carmen_analysis.name
  type         = "AAAA"
  ttl          = 300
  rrdatas = [
    "2001:4860:4802:32::15",
    "2001:4860:4802:34::15",
    "2001:4860:4802:36::15",
    "2001:4860:4802:38::15",
  ]
  project = var.project_id
}

# Google Search Console domain-ownership verification (required for Cloud Run
# domain mapping). Permanent — do not remove while the domain mapping exists.
resource "google_dns_record_set" "search_console_verification" {
  name         = "${var.domain}."
  managed_zone = google_dns_managed_zone.carmen_analysis.name
  type         = "TXT"
  ttl          = 300
  rrdatas      = ["\"google-site-verification=a-SRDz2SUXAEC3JFroqBwg4hWx6L6idvW05q9SK3rM4\""]
  project      = var.project_id
}
