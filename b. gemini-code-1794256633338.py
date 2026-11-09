import json
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

# Lista de acciones configuradas por el equipo de UNIPAZ
ACCIONES_CONFIG = [
    {
        "ticker": "ECOPETROL.CL",
        "nombre": "Ecopetrol",
        "nemotecnico": "ECOPETROL",
        "sector": "Petróleo y gas",
    },
    {
        "ticker": "PFCIBEST.CL",
        "nombre": "Grupo Cibest, preferencial",
        "nemotecnico": "PFCIBEST",
        "sector": "Financiero",
    },
    {
        "ticker": "CIBEST",  # Placeholder que requerirá verificación o fallback robusto
        "nombre": "Grupo Cibest, ordinaria",
        "nemotecnico": "CIBEST",
        "sector": "Financiero",
    },
    {
        "ticker": "ISA.CL",
        "nombre": "ISA",
        "nemotecnico": "ISA",
        "sector": "Energía e infraestructura",
    },
    {
        "ticker": "EXITO.CL",
        "nombre": "Grupo Éxito",
        "nemotecnico": "EXITO",
        "sector": "Comercio",
    },
    {
        "ticker": "PFAVAL.CL",
        "nombre": "Grupo Aval, preferencial",
        "nemotecnico": "PFAVAL",
        "sector": "Financiero",
    },
    {
        "ticker": "PFCORFICOL.CL",
        "nombre": "Libre elección 1 (Corficolombiana)",
        "nemotecnico": "PFCORFICOL",
        "sector": "Financiero / Inversiones",
    },
    {
        "ticker": "PFBCOLOM.CL",
        "nombre": "Libre elección 2 (Bancolombia)",
        "nemotecnico": "PFBCOLOM",
        "sector": "Financiero",
    },
]


def actualizar_datos():
    acciones_resultado = []
    errores = []

    for item in ACCIONES_CONFIG:
        ticker_sym = item["ticker"]
        try:
            print(f"Descargando datos para {ticker_sym}...")
            tk = yf.Ticker(ticker_sym)
            df = tk.history(period="3y", auto_adjust=False)

            if df is None or df.empty or len(df) < 5:
                raise ValueError("No se encontraron suficientes datos históricos.")

            # Último cierre y fecha
            ultimo_row = df.iloc[-1]
            precio = float(ultimo_row["Close"])
            fecha_str = df.index[-1].strftime("%Y-%m-%d")

            # Variación del día
            if len(df) >= 2:
                penultimo_cierre = float(df.iloc[-2]["Close"])
                variacion_dia_pct = (
                    (precio / penultimo_cierre) - 1.0
                ) * 100.0
            else:
                variacion_dia_pct = 0.0

            # Rentabilidad 1 año (ajustada por dividendos)
            # Buscar fecha de hace 1 año aproximadamente
            hace_1_anio = df.index[-1] - pd.DateOffset(years=1)
            df_1y = df[df.index >= hace_1_anio]
            if not df_1y.empty:
                adj_actual = float(ultimo_row["Adj Close"])
                adj_hace_1y = float(df_1y.iloc[0]["Adj Close"])
                rent_1a_pct = ((adj_actual / adj_hace_1y) - 1.0) * 100.0
            else:
                rent_1a_pct = 0.0

            # Rentabilidad anual promedio 3 años con fechas reales
            adj_primero = float(df.iloc[0]["Adj Close"])
            dias_transcurridos = (df.index[-1] - df.index[0]).days
            anos_transcurridos = dias_transcurridos / 365.25
            if anos_transcurridos > 0 and adj_primero > 0:
                rent_anual_3a_pct = (
                    (math_pow(adj_actual / adj_primero, 1.0 / anos_transcurridos)
                    - 1.0)
                    * 100.0
                )
            else:
                rent_anual_3a_pct = 0.0

            # Volatilidad anualizada (desviación estándar de rendimientos logarítmicos últimos 365 días * sqrt(252))
            df_vol = df[
                df.index >= df.index[-1] - pd.DateOffset(days=365)
            ].copy()
            if len(df_vol) > 1:
                df_vol["Log_Ret"] = (
                    df_vol["Adj Close"] / df_vol["Adj Close"].shift(1)
                ).apply(lambda x: 0 if x <= 0 else __import__("math").log(x))
                volatilidad_anual_pct = (
                    float(df_vol["Log_Ret"].std())
                    * __import__("math").sqrt(252)
                    * 100.0
                )
            else:
                volatilidad_anual_pct = 0.0

            # Caída máxima en 1 año (Maximum Drawdown)
            if len(df_vol) > 1:
                rolling_max = df_vol["Adj Close"].cummax()
                drawdown = (df_vol["Adj Close"] / rolling_max) - 1.0
                caida_maxima_1a_pct = float(drawdown.min()) * 100.0
            else:
                caida_maxima_1a_pct = 0.0

            # Rendimiento por dividendo (suma dividendos últimos 365 días / precio)
            if "Dividends" in df.columns:
                suma_divs = float(
                    df[
                        df.index >= df.index[-1]
                        - pd.DateOffset(days=365)
                    ]["Dividends"].sum()
                )
            else:
                suma_divs = 0.0
            dividendo_rend_pct = (
                (suma_divs / precio) * 100.0 if precio > 0 else 0.0
            )

            # Histórico mensual (último cierre de cada mes en los 3 años)
            df_mensual = df["Close"].resample("ME").last().dropna()
            historico = [
                {"mes": idx.strftime("%Y-%m"), "cierre": round(float(val), 2)}
                for idx, val in df_mensual.items()
            ]

            acciones_resultado.append(
                {
                    "ticker": ticker_sym,
                    "nombre": item["nombre"],
                    "nemotecnico": item["nemotecnico"],
                    "sector": item["sector"],
                    "moneda": "COP",
                    "precio": round(precio, 2),
                    "fecha": fecha_str,
                    "variacion_dia_pct": round(variacion_dia_pct, 2),
                    "rent_1a_pct": round(rent_1a_pct, 2),
                    "rent_anual_3a_pct": round(rent_anual_3a_pct, 2),
                    "volatilidad_anual_pct": round(volatilidad_anual_pct, 2),
                    "caida_maxima_1a_pct": round(caida_maxima_1a_pct, 2),
                    "dividendo_rend_pct": round(dividendo_rend_pct, 2),
                    "historico": historico,
                }
            )

        except Exception as e:
            print(f"Error procesando {ticker_sym}: {e}")
            errores.append({"ticker": ticker_sym, "motivo": str(e)})

    # Descargar TRM (USDCOP=X)
    trm_data = {"valor": 0, "fecha": "2026-11-09"}
    try:
        trm_tk = yf.Ticker("USDCOP=X")
        trm_df = trm_tk.history(period="5d")
        if not trm_df.empty:
            trm_data = {
                "valor": round(float(trm_df.iloc[-1]["Close"]), 2),
                "fecha": trm_df.index[-1].strftime("%Y-%m-%d"),
            }
    except Exception as e:
        print(f"No se pudo descargar la TRM: {e}")

    # Robustez: Si TODAS fallan, no sobrescribir datos.json
    if not acciones_resultado:
        print(
            "¡ADVERTENCIA CRÍTICA! Todas las acciones fallaron. No se sobrescribirá datos.json."
        )
        return

    output_data = {
        "actualizado": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "fuente": "Yahoo Finance (yfinance); rentabilidades con precio ajustado por dividendos",
        "trm": trm_data,
        "acciones": acciones_resultado,
        "errores": errores,
    }

    with open("datos.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(
        f"Actualización completada. Se actualizaron {len(acciones_resultado)} acciones. Errores: {len(errores)}."
    )


def math_pow(base, exp):
    return base**exp


if __name__ == "__main__":
    actualizar_datos()