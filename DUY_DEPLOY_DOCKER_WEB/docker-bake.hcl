variable "BASE_IMAGE" {
  default = "duy-yolo-pytorch:jetpack6"
}

variable "DOCKER_PLATFORM" {
  default = "linux/arm64"
}

variable "RUNTIME_TAG" {
  default = "duy-deploy-web:jetpack6"
}

group "default" {
  targets = ["runtime"]
}

target "runtime" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = [DOCKER_PLATFORM]
  tags       = [RUNTIME_TAG]
  args = {
    BASE_IMAGE = BASE_IMAGE
  }
}

target "workstation" {
  context    = "."
  dockerfile = "Dockerfile.workstation"
  platforms  = ["linux/amd64"]
  tags       = ["duy-deploy-web:workstation"]
  args = {
    BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
  }
}

target "portable-smoke" {
  context    = "."
  dockerfile = "Dockerfile.smoke"
  platforms  = ["linux/amd64"]
  tags       = ["duy-deploy-web:smoke"]
}
