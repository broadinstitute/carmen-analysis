###############################################################################
# Staging Cloud Run service                                                    #
#                                                                              #
# A second, deliberately permissive Cloud Run service used by CI to deploy a   #
# tagged revision per branch push, so engineers can test changes against the   #
# real container before merging.                                               #
#                                                                              #
# Differences from production (cloud_run.tf):                                  #
#   - ingress = ALL          → reachable directly at *.run.app                 #
#   - allUsers run.invoker   → no auth in front of it                          #
#   - lifecycle.ignore_changes on `template` → CI owns the live revisions      #
#                                                                              #
# Per-branch revisions are created by .github/workflows/docker.yml using       #
# `gcloud run deploy --tag <branch> --no-traffic`, which produces URLs of the  #
# form https://<branch>---<service>-<hash>.<region>.run.app — distinct per     #
# branch, no traffic shifting on the base URL.                                 #
###############################################################################

resource "google_cloud_run_v2_service" "staging" {
  name                = "${var.service_name}-staging"
  project             = var.project_id
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "${var.cloud_run_timeout_seconds}s"
    max_instance_request_concurrency = var.cloud_run_concurrency
    session_affinity                 = true

    scaling {
      min_instance_count = 0
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

  # CI is the source of truth for staging revisions. Terraform only owns the
  # service shell + IAM; new revisions arrive via `gcloud run deploy --tag ...`.
  lifecycle {
    ignore_changes = [
      template,
      client,
      client_version,
    ]
  }

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_v2_service_iam_member" "staging_public" {
  project  = google_cloud_run_v2_service.staging.project
  location = google_cloud_run_v2_service.staging.location
  name     = google_cloud_run_v2_service.staging.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
