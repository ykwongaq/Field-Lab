"""VideoSegmenter backend.

Layout:
    core/       configuration, errors, startup
    api/        FastAPI routers, dependencies, serializers, exception handlers
    schemas/    Pydantic request/response models
    domain/     framework-free mask/prompt logic (numpy + pycocotools only)
    projects/   project archive builder
    inference/  SAM 3 model wrappers (heavy, lazy imports)
    cli/        command-line entry points
"""
