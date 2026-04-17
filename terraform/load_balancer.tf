###############################################################################
# HTTPS External Application LB → Serverless NEG → Cloud Run, with IAP and a   #
# GCP-managed SSL certificate. Mirrors the pattern in sabeti-librechat-         #
# deployment but uses a GCP-managed cert (free, auto-renewing) rather than a    #
# BITS-issued cert.                                                             #
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

# --- Backend service (with IAP) -------------------------------------------- #

resource "google_iap_brand" "this" {
  project           = data.google_project.this.number
  support_email     = var.iap_support_email
  application_title = "CARMEN Analysis"
}

resource "google_iap_client" "app" {
  display_name = "${var.service_name}-iap-client"
  brand        = google_iap_brand.this.name
}

resource "google_compute_backend_service" "app" {
  name                  = "${var.service_name}-backend"
  project               = var.project_id
  protocol              = "HTTPS"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  enable_cdn            = false

  backend {
    group = google_compute_region_network_endpoint_group.app.id
  }

  iap {
    enabled              = true
    oauth2_client_id     = google_iap_client.app.client_id
    oauth2_client_secret = google_iap_client.app.secret
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# Grant configured principals access through IAP.
resource "google_iap_web_backend_service_iam_member" "users" {
  for_each            = toset(var.iap_members)
  project             = var.project_id
  web_backend_service = google_compute_backend_service.app.name
  role                = "roles/iap.httpsResourceAccessor"
  member              = each.value
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
