###############################################################################
# Artifact Registry repository that holds the carmen-analysis Docker image.    #
# The CI workflow (.github/workflows/docker.yml) pushes here.                  #
###############################################################################

resource "google_artifact_registry_repository" "carmen_analysis" {
  project       = var.project_id
  location      = var.region
  repository_id = var.gar_repository_id
  description   = "Container images for the CARMEN Analysis web app and CLI."
  format        = "DOCKER"

  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"
    most_recent_versions {
      keep_count = 20
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s"  # 7 days
    }
  }

  depends_on = [google_project_service.services]
}
