###############################################################################
# Required APIs                                                                 #
###############################################################################

resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
    "certificatemanager.googleapis.com",
    "iamcredentials.googleapis.com",
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

###############################################################################
# Static IP for the LB front-end                                                #
# Already created out-of-band so the user could file the BITS DNS request in    #
# parallel; we read it via a data source to avoid taking ownership / risking    #
# accidental destroy.                                                           #
###############################################################################

data "google_compute_global_address" "lb_ip" {
  name    = var.static_ip_name
  project = var.project_id
}
