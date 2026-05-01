###############################################################################
# Cloud Run service hosting the Streamlit web app                               #
###############################################################################

resource "google_cloud_run_v2_service" "app" {
  name                = var.service_name
  project             = var.project_id
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "${var.cloud_run_timeout_seconds}s"
    max_instance_request_concurrency = var.cloud_run_concurrency
    # Streamlit's session_state lives in the Python process; pin a browser
    # to one instance so XHRs and the websocket land together.
    session_affinity = true

    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }
    }
  }

  depends_on = [google_project_service.services]
}

# Public site: ingress=ALL above lets anyone reach the service over the
# *.run.app URL or the mapped FQDN; this binding lets anyone invoke. There's
# no auth tier — the data the pipeline processes lives on the user's laptop
# and nothing is retained server-side.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = google_cloud_run_v2_service.app.project
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
