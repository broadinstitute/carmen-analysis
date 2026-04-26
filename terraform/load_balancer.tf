###############################################################################
# HTTPS External Application LB → Serverless NEG → Cloud Run, with a GCP-       #
# managed SSL certificate. Public (no auth) — the LB just terminates TLS and    #
# fronts a Cloud Run service that's locked to ingress=INTERNAL_LB so direct     #
# *.run.app access is blocked.                                                  #
#                                                                               #
# Why no IAP: Google deprecated the IAP OAuth Admin API in July 2025, which     #
# orphaned the google_iap_brand / google_iap_client resources. The data this    #
# pipeline processes is on the user's laptop and nothing is retained server-    #
# side, so the auth tier wasn't load-bearing. A draft PR exists to migrate off  #
# the LB entirely (Cloud Run native domain mapping); see infra/cloudrun-domain- #
# mapping branch.                                                               #
###############################################################################

# --- Serverless NEG --------------------------------------------------------- #

resource "google_compute_region_network_endpoint_group" "app" {
  provider              = google-beta
  name                  = "${var.service_name}-neg"
  project               = var.project_id
  region                = var.region
  network_endpoint_type = "SERVERLESS"

  cloud_run {
    service = google_cloud_run_v2_service.app.name
  }
}

# --- Backend service ------------------------------------------------------- #

resource "google_compute_backend_service" "app" {
  name                  = "${var.service_name}-backend"
  project               = var.project_id
  protocol              = "HTTPS"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  enable_cdn            = false

  backend {
    group = google_compute_region_network_endpoint_group.app.id
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# --- URL map / target proxy ------------------------------------------------ #

resource "google_compute_url_map" "app" {
  name            = "${var.service_name}-url-map"
  project         = var.project_id
  default_service = google_compute_backend_service.app.id
}

# --- Managed SSL cert ------------------------------------------------------ #

resource "google_compute_managed_ssl_certificate" "app" {
  provider = google-beta
  name     = "${var.service_name}-cert"
  project  = var.project_id

  managed {
    domains = [var.domain]
  }
}

resource "google_compute_target_https_proxy" "app" {
  name             = "${var.service_name}-https-proxy"
  project          = var.project_id
  url_map          = google_compute_url_map.app.id
  ssl_certificates = [google_compute_managed_ssl_certificate.app.id]
}

resource "google_compute_global_forwarding_rule" "https" {
  name                  = "${var.service_name}-https-fr"
  project               = var.project_id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  port_range            = "443"
  target                = google_compute_target_https_proxy.app.id
  ip_address            = data.google_compute_global_address.lb_ip.id
}

# --- HTTP → HTTPS redirect ------------------------------------------------- #

resource "google_compute_url_map" "http_redirect" {
  name    = "${var.service_name}-http-redirect"
  project = var.project_id

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "app" {
  name    = "${var.service_name}-http-proxy"
  project = var.project_id
  url_map = google_compute_url_map.http_redirect.id
}

resource "google_compute_global_forwarding_rule" "http" {
  name                  = "${var.service_name}-http-fr"
  project               = var.project_id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  port_range            = "80"
  target                = google_compute_target_http_proxy.app.id
  ip_address            = data.google_compute_global_address.lb_ip.id
}
