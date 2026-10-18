# Celda para agregar AL FINAL de NB03 (Databricks). Mueve el endpoint a la versión @champion.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

w = WorkspaceClient()  # dentro de Databricks se autentica solo
version = client.get_model_version_by_alias(MODEL_NAME, "champion").version

w.serving_endpoints.update_config_and_wait(
    name="MLOPS_EAN",
    served_entities=[ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=str(version),
        workload_size="Small",
        scale_to_zero_enabled=True,
    )],
)
print(f"MLOPS_EAN ahora sirve la versión {version}")
