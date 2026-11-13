#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
PROYECTO DE AULA — MERCADO DE CAPITALES — UNIPAZ (Barrancabermeja, Colombia)
Firma Académica: Santander Asesores de Inversión
Archivo: actualizar_datos.py
Descripción:
    Descarga precios diarios de los últimos 3 años desde Yahoo Finance mediante
    yfinance y pandas para las principales acciones de la Bolsa de Valores de
    Colombia (BVC) y la TRM (USDCOP=X). Calcula métricas de rentabilidad,
    volatilidad, caída máxima, dividendos e histórico mensual, y guarda el
    resultado en 'datos.json' siguiendo la estructura exacta del proyecto.
Requisitos:
    Python 3.12+, yfinance, pandas, numpy
    Instalación: pip install yfinance pandas numpy
==============================================================================
"""

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ==============================================================================
# 1. CONFIGURACIÓN DE ACCIONES Y ARCHIVO DE SALIDA
# ==============================================================================
ARCHIVO_SALIDA = Path(__file__).resolve().parent / "datos.json"
TICKER_TRM = "USDCOP=X"

# Lista de 8 acciones de la BVC:
# (ticker_yahoo, nombre_empresa, nemotecnico_bvc, sector_economico)
ACCIONES_BVC: list[tuple[str, str, str, str]] = [
    ("ECOPETROL.CL", "Ecopetrol", "ECOPETROL", "Petróleo y gas"),
    ("PFCIBEST.CL", "Grupo Cibest preferencial", "PFCIBEST", "Financiero"),
    ("CIBEST.CL", "Grupo Cibest ordinaria", "CIBEST", "Financiero"),
    ("ISA.CL", "ISA", "ISA", "Energía e infraestructura"),
    ("EXITO.CL", "Grupo Éxito", "EXITO", "Comercio"),
    ("PFAVAL.CL", "Grupo Aval preferencial", "PFAVAL", "Financiero"),
    # Acción 7 escogida por el equipo asesor:
    ("GEB.CL", "Grupo Energía Bogotá", "GEB", "Energía"),
    # Acción 8 escogida por el equipo asesor:
    ("CELSIA.CL", "Celsia", "CELSIA", "Energía"),
]


# ==============================================================================
# 2. FUNCIONES AUXILIARES DE LIMPIEZA Y CÁLCULO FINANCIERO
# ==============================================================================
def normalizar_indice_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina la zona horaria del índice de fechas y ordena cronológicamente."""
    df_limpio = df.copy()
    if getattr(df_limpio.index, "tz", None) is not None:
        df_limpio.index = df_limpio.index.tz_localize(None)
    df_limpio.index = pd.to_datetime(df_limpio.index)
    df_limpio = df_limpio.sort_index()
    return df_limpio


def redondear_seguro(valor: float | np.floating | None, decimales: int = 2) -> float:
    """Convierte a float estándar de Python y redondea evitando NaN o infinitos."""
    if valor is None or pd.isna(valor) or not math.isfinite(float(valor)):
        return 0.0
    return round(float(valor), decimales)


def descargar_trm() -> dict[str, Any] | None:
    """
    Intenta descargar la TRM vigente desde Yahoo Finance ('USDCOP=X').
    Retorna {'valor': float, 'fecha': 'AAAA-MM-DD'} o None si falla.
    """
    try:
        ticker_obj = yf.Ticker(TICKER_TRM)
        hist = ticker_obj.history(period="10d", auto_adjust=False)

        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        hist = normalizar_indice_fechas(hist)
        cierres = hist["Close"].dropna()
        if cierres.empty:
            return None

        ultimo_cierre = float(cierres.iloc[-1])
        fecha_cierre = cierres.index[-1].strftime("%Y-%m-%d")

        if not math.isfinite(ultimo_cierre) or ultimo_cierre <= 0:
            return None

        return {
            "valor": redondear_seguro(ultimo_cierre, 2),
            "fecha": fecha_cierre,
        }
    except Exception as exc:
        print(f"[AVISO] No se pudo obtener la TRM ({TICKER_TRM}): {exc}")
        return None


def procesar_accion(
    ticker: str,
    nombre: str,
    nemotecnico: str,
    sector: str,
) -> dict[str, Any]:
    """
    Descarga 3 años de datos diarios para una acción y calcula todos los
    indicadores financieros requeridos por el aplicativo web.
    Lanza ValueError si los datos son insuficientes o inválidos.
    """
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="3y", auto_adjust=False)

    if hist is None or hist.empty:
        raise ValueError("Yahoo Finance devolvió una tabla vacía (sin cotizaciones).")

    columnas_requeridas = {"Close", "Adj Close"}
    if not columnas_requeridas.issubset(set(hist.columns)):
        raise ValueError(f"Faltan columnas requeridas en los datos: {hist.columns.tolist()}")

    hist = normalizar_indice_fechas(hist)

    # Filtrar filas donde Close y Adj Close sean válidos y mayores a cero
    df_valido = hist.dropna(subset=["Close", "Adj Close"]).copy()
    df_valido = df_valido[(df_valido["Close"] > 0) & (df_valido["Adj Close"] > 0)]

    if len(df_valido) < 2:
        raise ValueError("Menos de 2 días hábiles con precios válidos.")

    # 1. Precio actual y fecha del último cierre
    precio_actual = float(df_valido["Close"].iloc[-1])
    fecha_ultima: pd.Timestamp = df_valido.index[-1]
    fecha_str = fecha_ultima.strftime("%Y-%m-%d")

    # 2. Variación del día (%): último Close / Close anterior - 1
    precio_anterior = float(df_valido["Close"].iloc[-2])
    variacion_dia_pct = ((precio_actual / precio_anterior) - 1.0) * 100.0

    # 3. Rentabilidad 1 año (%): Adj Close último / Adj Close hace 1 año (fecha más cercana) - 1
    adj_ultimo = float(df_valido["Adj Close"].iloc[-1])
    fecha_objetivo_1a = fecha_ultima - pd.DateOffset(years=1)

    # Buscar la fecha más cercana en el índice a exactamente 1 año atrás
    diferencias_dias = np.abs((df_valido.index - fecha_objetivo_1a).days)
    idx_mas_cercano_1a = int(np.argmin(diferencias_dias))
    adj_hace_1a = float(df_valido["Adj Close"].iloc[idx_mas_cercano_1a])
    rent_1a_pct = ((adj_ultimo / adj_hace_1a) - 1.0) * 100.0

    # 4. Rentabilidad anual promedio 3 años (%):
    # (Adj Close último / Adj Close primero)^(1 / años transcurridos) - 1
    adj_primero = float(df_valido["Adj Close"].iloc[0])
    fecha_primera: pd.Timestamp = df_valido.index[0]
    dias_transcurridos = (fecha_ultima - fecha_primera).days

    if dias_transcurridos <= 0:
        raise ValueError("Rango de fechas insuficiente para calcular rentabilidad anualizada.")

    anios_transcurridos = dias_transcurridos / 365.25
    rent_anual_3a_pct = (
        math.pow(adj_ultimo / adj_primero, 1.0 / anios_transcurridos) - 1.0
    ) * 100.0

    # Subconjunto del último año (últimos 365 días calendario desde el último cierre)
    fecha_inicio_365d = fecha_ultima - timedelta(days=365)
    df_ultimo_anio = df_valido[df_valido.index >= fecha_inicio_365d].copy()
    if len(df_ultimo_anio) < 2:
        df_ultimo_anio = df_valido.tail(min(252, len(df_valido))).copy()

    # 5. Volatilidad anual (%): std de rendimientos logarítmicos diarios del último año * sqrt(252)
    retornos_log_1a = np.log(
        df_ultimo_anio["Adj Close"] / df_ultimo_anio["Adj Close"].shift(1)
    ).dropna()

    if len(retornos_log_1a) >= 2:
        volatilidad_anual_pct = float(retornos_log_1a.std(ddof=1) * math.sqrt(252)) * 100.0
    else:
        volatilidad_anual_pct = 0.0

    # 6. Caída máxima en 1 año (%): mayor caída desde un máximo previo (valor negativo)
    serie_adj_1a = df_ultimo_anio["Adj Close"]
    maximo_acumulado = serie_adj_1a.cummax()
    drawdowns = (serie_adj_1a / maximo_acumulado) - 1.0
    caida_maxima_1a_pct = float(drawdowns.min()) * 100.0
    if caida_maxima_1a_pct > 0:
        caida_maxima_1a_pct = 0.0

    # 7. Rendimiento por dividendo (%): suma de Dividends de los últimos 365 días / precio
    if "Dividends" in hist.columns:
        div_365d = hist.loc[hist.index >= fecha_inicio_365d, "Dividends"].fillna(0.0)
        suma_dividendos = float(div_365d[div_365d > 0].sum())
    else:
        suma_dividendos = 0.0

    dividendo_rend_pct = (suma_dividendos / precio_actual) * 100.0

    # 8. Histórico mensual de los 3 años: último Close de cada mes [{"mes": "AAAA-MM", "cierre": n}]
    cierres_mensuales = df_valido["Close"].resample("ME").last().dropna()
    historico_mensual = [
        {
            "mes": fecha_mes.strftime("%Y-%m"),
            "cierre": redondear_seguro(valor_cierre, 2),
        }
        for fecha_mes, valor_cierre in cierres_mensuales.items()
    ]

    return {
        "ticker": ticker,
        "nombre": nombre,
        "nemotecnico": nemotecnico,
        "sector": sector,
        "moneda": "COP",
        "precio": redondear_seguro(precio_actual, 2),
        "fecha": fecha_str,
        "variacion_dia_pct": redondear_seguro(variacion_dia_pct, 2),
        "rent_1a_pct": redondear_seguro(rent_1a_pct, 2),
        "rent_anual_3a_pct": redondear_seguro(rent_anual_3a_pct, 2),
        "volatilidad_anual_pct": redondear_seguro(volatilidad_anual_pct, 2),
        "caida_maxima_1a_pct": redondear_seguro(caida_maxima_1a_pct, 2),
        "dividendo_rend_pct": redondear_seguro(dividendo_rend_pct, 2),
        "historico": historico_mensual,
    }


# ==============================================================================
# 3. FUNCIÓN PRINCIPAL DE ORQUESTACIÓN Y GUARDADO SEGURO
# ==============================================================================
def main() -> int:
    """
    Ejecuta la actualización completa de datos bursátiles y guarda datos.json.
    Siempre retorna 0 para no interrumpir flujos automatizados (GitHub Actions).
    """
    print("=" * 72)
    print("SANTANDER ASESORES DE INVERSIÓN — UNIPAZ")
    print("Iniciando descarga y actualización de datos reales desde Yahoo Finance...")
    print("=" * 72)

    # Descargar TRM
    print(f"Consultando TRM ({TICKER_TRM})...")
    trm_info = descargar_trm()
    if trm_info is not None:
        print(f"  -> TRM obtenida: {trm_info['valor']} COP ({trm_info['fecha']})")
    else:
        print("  -> TRM no disponible; se registrará como null.")

    acciones_exitosas: list[dict[str, Any]] = []
    errores: list[dict[str, str]] = []

    # Procesar cada acción de manera independiente y tolerante a fallos
    for ticker, nombre, nemotecnico, sector in ACCIONES_BVC:
        print(f"Descargando {ticker} ({nombre})...")
        try:
            datos_accion = procesar_accion(ticker, nombre, nemotecnico, sector)
            acciones_exitosas.append(datos_accion)
            print(
                f"  [OK] Precio: {datos_accion['precio']} COP | "
                f"Rent 1A: {datos_accion['rent_1a_pct']}% | "
                f"Vol: {datos_accion['volatilidad_anual_pct']}%"
            )
        except Exception as exc:
            motivo_error = str(exc).strip() or exc.__class__.__name__
            errores.append({"ticker": ticker, "motivo": motivo_error})
            print(f"  [ERROR] {ticker}: {motivo_error}")

    print("-" * 72)

    # Protección: si TODAS las acciones fallaron, no sobrescribir datos.json
    if len(acciones_exitosas) == 0:
        print(
            "[AVISO CRÍTICO] Ninguna acción pudo descargarse correctamente. "
            "No se sobrescribirá 'datos.json' para conservar la versión anterior."
        )
    else:
        marca_tiempo_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        estructura_json = {
            "actualizado": marca_tiempo_utc,
            "fuente": "Yahoo Finance (yfinance); rentabilidades con precio ajustado por dividendos",
            "trm": trm_info,
            "acciones": acciones_exitosas,
            "errores": errores,
        }

        try:
            with ARCHIVO_SALIDA.open("w", encoding="utf-8") as archivo:
                json.dump(estructura_json, archivo, ensure_ascii=False, indent=2)
            print(f"[GUARDADO] Archivo actualizado exitosamente en: {ARCHIVO_SALIDA}")
        except Exception as exc:
            print(f"[ERROR DE ESCRITURA] No se pudo guardar {ARCHIVO_SALIDA}: {exc}")

    # Resumen final de ejecución
    print("\nRESUMEN DE EJECUCIÓN:")
    print(f"  • Acciones actualizadas correctamente: {len(acciones_exitosas)} de {len(ACCIONES_BVC)}")
    if errores:
        print(f"  • Acciones con error ({len(errores)}):")
        for err in errores:
            print(f"      - {err['ticker']}: {err['motivo']}")
    else:
        print("  • Acciones con error: 0 (todas exitosas)")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
