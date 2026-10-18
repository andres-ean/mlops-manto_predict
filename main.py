# API FastAPI que recibe los datos de una máquina y consulta el endpoint MLOPS_EAN de Databricks
# Modelo: workspace.default.mantenimiento_predictivo (@champion) -> predice si la máquina falla
import math
import os
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Credenciales: se inyectan como variables de entorno en Azure (nunca escritas en el código)
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

app = FastAPI(title="API MLOps EAN", description="Predicción de falla en máquinas (mantenimiento predictivo)")

# Permite que el frontend (index.html) llame a la API desde otro dominio
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mismo orden y nombres que FEATURES en NB02 (firma del modelo en MLflow)
FEATURES = [
    "tipo_cod", "temp_aire_k", "temp_proceso_k", "velocidad_rpm",
    "torque_nm", "desgaste_min", "dif_temperatura", "potencia_w",
]
TIPO_COD = {"L": 0, "M": 1, "H": 2}  # igual que en NB01


# Datos crudos que envía el usuario (6 campos). Las variables derivadas las calcula la API.
class Maquina(BaseModel):
    tipo: Literal["L", "M", "H"]
    temp_aire_k: float = Field(..., ge=250, le=350)
    temp_proceso_k: float = Field(..., ge=250, le=400)
    velocidad_rpm: float = Field(..., ge=0, le=5000)
    torque_nm: float = Field(..., ge=0, le=150)
    desgaste_min: float = Field(..., ge=0, le=500)


def construir_registro(m: Maquina) -> dict:
    # Misma ingeniería de variables que NB01 (si cambia allá, cambia aquí)
    return {
        "tipo_cod": float(TIPO_COD[m.tipo]),
        "temp_aire_k": m.temp_aire_k,
        "temp_proceso_k": m.temp_proceso_k,
        "velocidad_rpm": m.velocidad_rpm,
        "torque_nm": m.torque_nm,
        "desgaste_min": m.desgaste_min,
        "dif_temperatura": m.temp_proceso_k - m.temp_aire_k,
        "potencia_w": m.torque_nm * m.velocidad_rpm * 2 * math.pi / 60,
    }


@app.get("/health")
def health():
    # Verifica que la API está viva y que tiene configuradas las credenciales
    return {"status": "ok", "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN)}


@app.post("/predecir")
def predecir(m: Maquina):
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN")

    registro = construir_registro(m)
    payload = {"dataframe_records": [{k: registro[k] for k in FEATURES}]}  # formato exigido por Databricks

    try:
        r = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload,
            timeout=120,  # margen para el arranque en frío del endpoint
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Databricks tardó demasiado (arranque en frío). Intente de nuevo.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"No se pudo contactar a Databricks: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error de Databricks: {r.text}")

    pred = int(r.json()["predictions"][0])  # 1 = falla, 0 = sin falla
    return {
        "prediccion": pred,
        "resultado": "Falla probable ⚠️" if pred == 1 else "Sin falla ✅",
        "datos_recibidos": m.model_dump(),
        "variables_calculadas": {
            "tipo_cod": registro["tipo_cod"],
            "dif_temperatura": round(registro["dif_temperatura"], 2),
            "potencia_w": round(registro["potencia_w"], 1),
        },
    }
