###############################################################################
# Cloud Run service hosting the Streamlit web app                               #
###############################################################################

resource "google_cloud_run_v2_service" "app" {
  name                = var.service_name
  project             = var.project_id
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email

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

# IAP fronts the LB; only the IAP service principal needs run.invoker on the
# Cloud Run service itself.
resource "google_cloud_run_v2_service_iam_member" "iap_invoker" {
  project  = google_cloud_run_v2_service.app.project
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-iap.iam.gserviceaccount.com"
}

data "google_project" "this" {
  project_id = var.project_id
}
