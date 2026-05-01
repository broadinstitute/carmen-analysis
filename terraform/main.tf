###############################################################################
# Required APIs                                                                 #
###############################################################################

resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iamcredentials.googleapis.com",
    "dns.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

###############################################################################
# Service account that the Cloud Run service runs as                            #
###############################################################################

resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-run"
  display_name = "Runtime SA for ${var.service_name} Cloud Run service"
  project      = var.project_id
}
