import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Configuración de la página
st.set_page_config(page_title="Evaluación de Crédito - Ecuador", layout="wide")


# ---------------------------------------------------------
# 1. CARGA DE DATOS CON RUTA ABSOLUTA AUTOMÁTICA
# ---------------------------------------------------------
@st.cache_resource
def entrenar_modelo_real():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    nombres_posibles = [
        "Base_datos_práctica.csv",
        "Base_datos_practica.csv",
        "BASE_DATOS_PRACTICA.csv",
        "Base_datos_práctica.csv.csv",
        "Base_datos_practica.csv.csv"
    ]

    archivo_detectado = None
    for nombre in nombres_posibles:
        ruta_completa = os.path.join(directorio_actual, nombre)
        if os.path.exists(ruta_completa):
            archivo_detectado = ruta_completa
            break

    if not archivo_detectado:
        for nombre in nombres_posibles:
            if os.path.exists(nombre):
                archivo_detectado = nombre
                break

    if archivo_detectado:
        try:
            df = pd.read_csv(archivo_detectado, sep=';', decimal=',')
            if 'Aprobado' not in df.columns:
                df = pd.read_csv(archivo_detectado, sep=',', decimal='.')

            X = df[['Hombre', 'Mujer', 'Ingresos', 'Casados', 'Ratio_Gastos_Ingreso', 'Ratio_Deuda_Ingreso']].dropna()
            y = df.loc[X.index, 'Aprobado']

            modelo = LogisticRegression(max_iter=1000)
            modelo.fit(X, y)
            return modelo, df
        except Exception as e:
            return None, None
    return None, None


modelo, df_historico = entrenar_modelo_real()

# Título Principal
st.title("📊 Sistema Predictivo y de Riesgo de Crédito - Normativa Ecuador")
st.markdown(
    "Análisis de riesgo financiero parametrizado bajo los segmentos regulados del sistema bancario y cooperativo ecuatoriano.")

# ---------------------------------------------------------
# INDICACIONES DE USO (CÓMO FUNCIONA EL SIMULADOR)
# ---------------------------------------------------------
with st.expander("ℹ️ ¿Cómo funciona este simulador? (Indicaciones de Uso)", expanded=False):
    st.markdown("""
    Este simulador automatiza la evaluación de solicitudes de crédito combinando **reglas de política financiera (normativa de Ecuador)** y un **modelo estadístico predictivo (Regresión Logística)** basado en datos históricos.

    ### 🛠️ Pasos para realizar una simulación:
    1. **Paso 1 (Datos del Cliente):** Selecciona el sexo, estado civil y el **Segmento de Crédito** que deseas evaluar (cada segmento tiene topes de riesgo diferentes según los organismos de control).
    2. **Paso 2 (Información Financiera):** Digita los ingresos netos mensuales, el monto que solicita el cliente y el valor comercial del bien base.
    3. **Paso 3 (Ratios de Endeudamiento):** 
       * **S45 (Ratio Cuota/Ingreso):** Qué porcentaje del sueldo se destinará a cubrir la cuota de este nuevo crédito.
       * **S46 (Ratio Deuda Total/Ingreso):** Qué porcentaje del sueldo absorben **todas** las deudas consolidadas del cliente.

    ### 📊 ¿Cómo se determina el Dictamen Final?
    El sistema analiza tres filtros lógicos en tiempo real:
    * **Filtro 1: Políticas de Riesgo:** Verifica que el endeudamiento total (`S46`) no supere los techos prudenciales (ej. máximo 40% en créditos VIP o Consumo).
    * **Filtro 2: Modelo Estadístico:** La Regresión Logística evalúa el perfil macro de la solicitud para estimar probabilísticamente la viabilidad de la operación.
    * **Filtro 3: Lógica de Mitigantes:** Si el perfil inicial incumple las políticas estándar pero activas la casilla de **Garantías Extra**, el sistema recalcula el indicador *LTV (Loan-to-Value)*. Si el nivel de colateral compensa la operación bajando el LTV a rangos seguros, el crédito se "rescata" bajo un dictamen condicional.
    """)

st.markdown("---")

# Pestañas de la aplicación
tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Simulador de Aprobación",
    "📈 Score Crediticio & Semáforo",
    "📝 Diagnóstico Automático: 5 C",
    "📋 Conclusiones y Recomendaciones"
])

with tab1:
    st.header("Formulario del Solicitante")
    st.write("Modifica los valores para calcular la aprobación e indicadores en tiempo real:")

    # FILA 1: DATOS GENERALES Y SEGMENTO
    col1, col2, col3 = st.columns(3)

    with col1:
        sexo = st.selectbox("S15: Sexo del Solicitante", options=["Hombre", "Mujer"])
    with col2:
        estado_civil = st.selectbox("S34: Estado Civil", options=["Casado/a", "Soltero/a / Otro"])
    with col3:
        tipo_credito = st.selectbox(
            "💼 Segmento de Crédito (Ecuador)",
            options=[
                "Crédito Inmobiliario (Vivienda de Interés Público - VIP)",
                "Crédito Inmobiliario Estándar (Vivienda / Terrenos)",
                "Crédito de Consumo (Ordinario / Prioritario)",
                "Microcrédito Minorista (Negocios Pequeños / Autoempleo)",
                "Microcrédito de Acumulación (Simple / Ampliada)",
                "Crédito Comercial / Productivo Pymes"
            ]
        )

    st.markdown("---")

    # FILA 2: DATOS FINANCIEROS CUANTITATIVOS (Estructura corregida con valores base válidos)
    col4, col5, col6 = st.columns(3)

    with col4:
        ingresos = st.number_input("S17: Ingresos Netos Mensuales ($)", min_value=1.0, max_value=200000.0, value=750.0,
                                   step=50.0)
        monto_solicitado = st.number_input("💰 Monto del Préstamo Solicitado ($)", min_value=100.0, max_value=1000000.0,
                                           value=15000.0, step=1000.0)

    with col5:
        dti_vivienda = st.number_input("S45: Ratio Cuota Crédito / Ingreso Mensual (%)", min_value=0.0,
                                       max_value=1000.0, value=25.0, step=1.0)

    with col6:
        dti_total = st.number_input("S46: Ratio Deuda Total / Ingreso Neto (%)", min_value=0.0, max_value=1000.0,
                                    value=35.0, step=1.0)
        valor_vivienda = st.number_input("🏠 Valor de Mercado del Bien Base / Proyecto ($)", min_value=100.0,
                                         max_value=1000000.0, value=20000.0, step=1000.0)

    st.markdown("---")

    # ---------------------------------------------------------
    # EVALUACIÓN DE MITIGANTES Y GARANTÍAS EXTRA
    # ---------------------------------------------------------
    st.markdown("#### 🛡️ Evaluación de Mitigantes y Garantías Extra")

    col_gar1, col_gar2 = st.columns(2)

    with col_gar1:
        garantia_extra = st.checkbox(
            "¿El cliente ofrece una garantía prendaria, hipotecaria adicional o un Co-signatario sólido?")

    with col_gar2:
        if garantia_extra:
            tipo_garantia = st.selectbox("Tipo de Garantía Adicional",
                                         options=["Hipoteca de Bien Raíz Extra", "Prenda Comercial / Vehículo",
                                                  "Garantía Personal / Garante Solidario",
                                                  "Depósito a Plazo Fijo (DPF)"])
            valor_garantia = st.number_input("Valor Comercial de la Garantía Extra / Respaldo ($)", min_value=0.0,
                                             max_value=500000.0, value=5000.0, step=500.0)
        else:
            valor_garantia = 0.0
            tipo_garantia = "Ninguna"

    # Cálculo del Cobertura LTV Técnico
    valor_cobertura_total = valor_vivienda + valor_garantia
    ltv_calculado = (monto_solicitado / valor_cobertura_total) * 100 if valor_cobertura_total > 0 else 0

    st.markdown(
        f"**Indicador Técnico de Cobertura (LTV Real):** `{ltv_calculado:.2f}%` (El crédito representa el {ltv_calculado:.2f}% de las garantías presentadas).")

    st.markdown("---")

    # ---------------------------------------------------------
    # PREDICCIÓN DINÁMICA CON AJUSTES ECUATORIANOS
    # ---------------------------------------------------------
    st.subheader("💡 Dictamen de la Solicitud (S7: Tipo de Acción)")

    aprobado_final = False

    # Parámetros de políticas basados en el segmento de Ecuador
    if "VIP" in tipo_credito:
        umbral_dti = 40.0
        max_ltv = 95.0
    elif "Inmobiliario Estándar" in tipo_credito:
        umbral_dti = 45.0
        max_ltv = 80.0
    elif "Consumo" in tipo_credito:
        umbral_dti = 40.0
        max_ltv = 150.0
    elif "Microcrédito" in tipo_credito:
        umbral_dti = 50.0
        max_ltv = 100.0
    else:
        umbral_dti = 45.0
        max_ltv = 85.0

    # Lógica de aprobación base según políticas institucionales
    if dti_total < umbral_dti and dti_vivienda < 38.0 and (ltv_calculado <= max_ltv or "Consumo" in tipo_credito):
        aprobado_base = True
    else:
        aprobado_base = False

    if modelo is not None:
        es_hombre = 1 if sexo == "Hombre" else 0
        es_mujer = 1 if sexo == "Mujer" else 0
        es_casado = 1 if estado_civil == "Casado/a" else 0

        # Ajustar escala para que coincida con los máximos del archivo original
        ingresos_escala = ingresos / 1000.0 if df_historico['Ingresos'].max() < 1000 else ingresos
        entrada = np.array([[es_hombre, es_mujer, ingresos_escala, es_casado, dti_vivienda, dti_total]])
        prediccion = modelo.predict(entrada)[0]
        probabilidad_final = modelo.predict_proba(entrada)[0][1]

        if prediccion == 1 or probabilidad_final > 0.05 or aprobado_base:
            aprobado_final = True
    else:
        aprobado_final = aprobado_base

    # Despliegue dinámico y visual del veredicto
    if aprobado_final:
        st.success(
            f"🎉 **CRÉDITO APROBADO** - El solicitante califica para las políticas del segmento: **{tipo_credito}**.")
    elif not aprobado_final and garantia_extra and ltv_calculado < 70.0:
        st.warning(
            f"🔄 **CRÉDITO APROBADO CON MITIGANTES** - La estructura financiera inicial presentaba debilidades para el segmento **{tipo_credito}**. Sin embargo, la asignación de un respaldo tipo '{tipo_garantia}' por **${valor_garantia:,.2f}** disminuye el LTV global al `{ltv_calculado:.2f}%`, haciendo viable el riesgo institucional.")
    else:
        st.error(
            f"❌ **CRÉDITO RECHAZADO** - El cliente excede las capacidades del segmento **{tipo_credito}**. Las garantías o el nivel de apalancamiento de deudas no se ajustan a las políticas de riesgo actuales.")

# ---------------------------------------------------------
# PESTAÑA: SCORE CREDITICIO CON VELOCÍMETRO VISUAL
# ---------------------------------------------------------
with tab2:
    st.header("📈 Tacómetro de Score Crediticio de Buró")

    score_cliente = st.slider("Seleccione el Score del Cliente:", min_value=300, max_value=850, value=700, step=5)

    if score_cliente < 500:
        rango, color = "BAJO", "🔴"
    elif score_cliente < 650:
        rango, color = "REGULAR", "🟠"
    elif score_cliente < 750:
        rango, color = "BUENO", "🟡"
    else:
        rango, color = "EXCELENTE", "🟢"

    st.markdown(f"### {color} Rango Actual del Buró: **{rango}** ({score_cliente} Puntos)")

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.barh(1, np.pi / 4, left=3 * np.pi / 4, color='#e74c3c', alpha=0.9, height=0.4)
    ax.barh(1, np.pi / 4, left=2 * np.pi / 4, color='#e67e22', alpha=0.9, height=0.4)
    ax.barh(1, np.pi / 4, left=np.pi / 4, color='#f1c40f', alpha=0.9, height=0.4)
    ax.barh(1, np.pi / 4, left=0, color='#2ecc71', alpha=0.9, height=0.4)

    proporcion_score = (score_cliente - 300) / (850 - 300)
    angulo_aguja = np.pi - (proporcion_score * np.pi)
    ax.annotate('', xy=(angulo_aguja, 1.1), xytext=(0, 0),
                arrowprops=dict(facecolor='black', width=3, headwidth=8, shrink=0.1))

    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.2)
    ax.set_frame_on(False)
    ax.set_xticks([0, np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(['850\n(Exc)', '750\n(Bueno)', '650\n(Reg)', '500\n(Bajo)', '300'])
    ax.set_yticklabels([])
    ax.grid(False)
    st.pyplot(fig)

# ---------------------------------------------------------
# IDENTIFICACIÓN AUTOMÁTICA DE LAS 5 C
# ---------------------------------------------------------
with tab3:
    st.header("Análisis Cualitativo Automatizado: Las 5 C")

    if dti_total <= 35:
        capacidad_auto = "🟢 Alta Capacidad - Deudas controladas."
    elif dti_total <= 45:
        capacidad_auto = "🟡 Capacidad Moderada - Límites estándar del sector."
    else:
        capacidad_auto = "🔴 Capacidad Crítica - Alto sobreendeudamiento normativo."

    if garantia_extra:
        colateral_auto = f"🟢 Robusto - Cobertura extendida mediante '{tipo_garantia}' de ${valor_garantia:,.2f}."
        capital_auto = f"🟢 Estructura patrimonial sólida. LTV seguro del {ltv_calculado:.1f}%."
    else:
        if dti_vivienda <= 30:
            colateral_auto = "🟢 Conforme - El valor base cubre el préstamo."
            capital_auto = "🟢 Aporte básico líquido suficiente."
        else:
            colateral_auto = "🟡 Ajustado / Cobertura patrimonial al límite."
            capital_auto = "🔴 Capital expuesto al riesgo del producto."

    if score_cliente >= 650:
        caracter_auto = "🟢 Reputación Confiable - Historial de pago saludable."
        condiciones_auto = f"🟢 Viable - Conforme a los lineamientos para: {tipo_credito}."
    else:
        caracter_auto = "🔴 Perfil de Riesgo Alto - Central de riesgos con novedades pasadas."
        condiciones_auto = f"🔴 Restringido - Alerta política para colocar en {tipo_credito}."

    st.markdown("### 📋 Reporte Integrado de las 5 C")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**1. Carácter:** {caracter_auto}")
        st.info(f"**2. Capacidad:** {capacidad_auto}")
        st.info(f"**3. Capital:** {capital_auto}")
    with c2:
        st.info(f"**4. Colateral:** {colateral_auto}")
        st.info(f"**5. Condiciones:** {condiciones_auto}")

# ---------------------------------------------------------
# CONCLUSIONES Y RECOMENDACIONES
# ---------------------------------------------------------
with tab4:
    st.header("📌 Conclusiones y Recomendaciones Finales")

    st.subheader("Conclusiones")
    try:
        if aprobado_final:
            if garantia_extra and (dti_total > umbral_dti or score_cliente < 650):
                st.markdown(
                    f"* **Excepción por Colateral:** Solicitud aprobada condicionalmente en el segmento **{tipo_credito}** gracias a un colateral extra de **${valor_garantia:,.2f}**.")
            else:
                st.markdown(
                    f"* **Aprobación Regular:** El cliente cumple perfectamente el perfil idóneo para la línea de **{tipo_credito}** con un indicador de cobertura patrimonial ($LTV$) del `{ltv_calculado:.2f}%`.")
        else:
            st.markdown(
                f"* **Rechazo Técnico:** El perfil evaluado supera los límites consolidados para la cartera de **{tipo_credito}**.")
    except:
        st.markdown(f"* **Evaluación en curso:** Ingrese datos válidos en los controles.")
