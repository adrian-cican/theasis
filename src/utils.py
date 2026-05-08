"""Shared utility functions."""

TAG_INFO = {
    "FT-201": {"label": "Biogas inlet flow", "unit": "Nm3/h"},
    "FT-801": {"label": "Biomethane outlet flow", "unit": "Nm3/h"},
    "FT-901": {"label": "Exhaust gas flow", "unit": "Nm3/h"},
    "Mode": {"label": "Run mode", "unit": "integer"},
    "PT-901": {"label": "Gas line inlet vacuum pressure", "unit": "bar"},
    "PT-902": {"label": "Gas line outlet vacuum pressure", "unit": "bar"},
    "PT-903": {"label": "Vacuum pressure in oil separator", "unit": "bar"},
    "TT-901": {"label": "Oil separator gas temperature", "unit": "°C"},
    "TT-902": {"label": "Discharge 1 temperature", "unit": "°C"},
    "TT-903": {"label": "Discharge 2 temperature", "unit": "°C"},
    "TT-904": {"label": "Oil temperature", "unit": "°C"},
    "VSD-901_CORRENT": {"label": "Vacuum pump current", "unit": "A"},
    "VSD-901_POWER": {"label": "Vacuum pump power", "unit": "W"},
    "VSD-901_RPM": {"label": "Vacuum pump rotation", "unit": "RPM"},
    "VSD-901_SPEED": {"label": "Vacuum pump speed", "unit": "%"},
    "LS-901": {"label": "Oil level sensor", "unit": "true/false"},
}

pressure_cols = ["PT-901", "PT-902", "PT-903"]
temp_cols = ["TT-901", "TT-902", "TT-903", "TT-904"]
flow_cols = ["FT-201", "FT-801", "FT-901"]

pump_oil_level_cols = ["LS-901"]
pump_current_cols = ["VSD-901_CORRENT"]
pump_power_cols = ["VSD-901_POWER"]
pump_rpm_cols = ["VSD-901_RPM"]
pump_speed_cols = ["VSD-901_SPEED"]




def inject_realistic_fault(df, anchor_col, target_value, start_perc=0.7):
    df_faulty = df.copy()
    n = len(df_faulty)
    start_idx = int(n * start_perc)
    
    # 1. Calcular correlações da coluna âncora com as outras
    corr_matrix = df.corr()[anchor_col]
    
    # 2. Calcular o Delta necessário para o âncora chegar ao target (ex: Warning level)
    current_val = df_faulty[anchor_col].iloc[start_idx]
    total_delta = target_value - current_val
    
    # 3. Gerar a rampa de falha
    ramp = np.linspace(0, total_delta, n - start_idx)
    
    # 4. Injetar no âncora e propagar para os outros baseado na correlação
    for col in df_faulty.columns:
        if col in TAG_INFO: # Apenas sensores reais
            r = corr_matrix[col]
            # Só propagamos se a correlação for minimamente relevante (> 0.3)
            if abs(r) > 0.3:
                # O sinal (+ ou -) da correlação garante que se um sobe, 
                # o outro sobe ou desce conforme o comportamento real
                df_faulty.iloc[start_idx:, df_faulty.columns.get_loc(col)] += ramp * r
                
    df_faulty['is_anomaly'] = 0
    df_faulty.iloc[start_idx:, df_faulty.columns.get_loc('is_anomaly')] = 1
    
    return df_faulty

# Exemplo: Simular aquecimento excessivo (TT-904 a chegar ao Warning de 90ºC)
#dict_sessions[17] = inject_realistic_fault(dict_sessions[17], anchor_col="TT-904", target_value=90)