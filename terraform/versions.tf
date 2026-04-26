terraform {
  # 1.5 is what Homebrew currently ships (Hashicorp's BUSL change blocked
  # later versions from the formula). Bump only if we actually need a newer
  # feature — none of this stack uses 1.6+ syntax.
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}
