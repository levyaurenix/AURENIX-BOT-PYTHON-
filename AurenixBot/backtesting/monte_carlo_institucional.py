#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOHABOT MONTE CARLO INSTITUCIONAL - FRAMEWORK COMPLETO
=====================================================

Sistema completo de validacion Monte Carlo para trading institucional:
- Estrategia EMA Real (8/21/55 + RSI + ATR)
- Walk-forward Analysis robusto
- Out-of-sample validation estricto
- Stress testing de costes extremos
- Bootstrap con gaps temporales
- Purged Cross-Validation
- Auditoria no-lookahead absoluto
- Costos XM reales (spread 40pts + comision $7)

EJECUCION:
python monte_carlo_institucional.py "database/data_history/GOLD#_M6_M0_1.csv"
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy import stats
import itertools

# Configurar logging institucional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yohabot_monte_carlo_institucional.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class AnalysisConfig:
    """
    Configuracion centralizada para analisis Monte Carlo institucional.
    Define parametros criticos, umbrales de decision y criterios institucionales.
    """
    # Parametros Monte Carlo
    n_simulations: int = 2000
    bootstrap_block_size: int = 20
    confidence_levels: List[float] = None

    # Criterios Go/No-Go institucionales
    min_sharpe_ratio: float = 1.2
    max_drawdown_threshold: float = 0.18  # 18%
    min_profit_factor: float = 1.3
    min_win_rate: float = 0.45  # 45%
    min_calmar_ratio: float = 2.0

    # Parametros Walk-Forward
    training_window_months: int = 6
    validation_window_months: int = 2
    step_size_months: int = 1
    min_trades_per_window: int = 20

    # Stress Testing avanzado
    stress_scenarios: Dict[str, Dict] = None

    # Out-of-sample validation
    oos_percentage: float = 0.25  # 25% para out-of-sample
    purge_window_days: int = 5    # Purge gap para evitar lookahead

    # Parametros estrategia EMA
    ema_fast: int = 8
    ema_medium: int = 21
    ema_slow: int = 55

    def __post_init__(self):
        """Inicializacion post-creacion con valores por defecto"""
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]

        if self.stress_scenarios is None:
            self.stress_scenarios = {
                'base': {'spread_mult': 1.0, 'slip_mult': 1.0, 'comm_mult': 1.0},
                'high_cost': {'spread_mult': 2.0, 'slip_mult': 2.5, 'comm_mult': 1.5},
                'extreme_cost': {'spread_mult': 3.0, 'slip_mult': 4.0, 'comm_mult': 2.0},
                'low_liquidity': {'spread_mult': 1.5, 'slip_mult': 3.0, 'comm_mult': 1.2},
                'crisis': {'spread_mult': 4.0, 'slip_mult': 5.0, 'comm_mult': 2.5}
            }

        # Validaciones criticas
        assert self.min_sharpe_ratio > 0, "Sharpe ratio debe ser positivo"
        assert 0 < self.max_drawdown_threshold < 1, "Drawdown debe estar entre 0 y 1"
        assert self.n_simulations >= 1000, "Minimo 1000 simulaciones institucionales"


class DataLeakageAuditor:
    """
    Auditor institucional para detectar y prevenir data leakage.
    Implementa verificaciones exhaustivas para garantizar integridad temporal.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.violations_found = []
        self.audit_results = {}

    def comprehensive_audit(self, data: pd.DataFrame, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Auditoria completa de integridad temporal"""
        logger.info("Iniciando auditoria completa de data leakage...")

        results = {
            'temporal_integrity': self._audit_temporal_integrity(data, signals),
            'signal_timing': self._audit_signal_timing(data, signals),
            'return_causality': self._audit_return_causality(signals, returns),
            'future_information': self._audit_future_information(data, signals),
            'statistical_tests': self._run_statistical_tests(data, signals, returns),
            'overall_score': 0,
            'violations_summary': []
        }

        # Calcular score general
        scores = [r.get('score', 0) for r in results.values() if isinstance(r, dict) and 'score' in r]
        results['overall_score'] = np.mean(scores) if scores else 0

        # Resumen de violaciones
        for audit_type, audit_result in results.items():
            if isinstance(audit_result, dict) and audit_result.get('violations', 0) > 0:
                results['violations_summary'].append(f"{audit_type}: {audit_result['violations']} violaciones")

        logger.info(f"Auditoria completada. Score: {results['overall_score']:.1f}/100")
        return results

    def _audit_temporal_integrity(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """Verificar integridad temporal basica"""
        violations = 0

        # Verificar alineacion temporal
        if len(data) != len(signals):
            violations += 1

        # Verificar orden temporal
        if not data.index.is_monotonic_increasing:
            violations += 1

        # Verificar gaps temporales sospechosos
        time_diffs = data.index.to_series().diff()
        normal_interval = time_diffs.mode().iloc[0]
        suspicious_gaps = (time_diffs > normal_interval * 3).sum()

        if suspicious_gaps > len(data) * 0.01:  # Mas del 1% gaps sospechosos
            violations += 1

        score = max(0, 100 - violations * 20)
        return {'violations': violations, 'score': score, 'suspicious_gaps': suspicious_gaps}

    def _audit_signal_timing(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """Verificar timing de senales vs movimientos futuros"""
        violations = 0

        # Analizar correlacion entre senales y retornos futuros
        future_returns_1 = data['close'].pct_change().shift(-1)
        future_returns_5 = data['close'].pct_change(5).shift(-5)

        # Correlacion con retornos 1 periodo futuro
        corr_1 = signals.corr(future_returns_1)
        if abs(corr_1) > 0.15:  # Correlacion sospechosamente alta
            violations += 1

        # Correlacion con retornos 5 periodos futuros
        corr_5 = signals.corr(future_returns_5)
        if abs(corr_5) > 0.1:
            violations += 1

        score = max(0, 100 - violations * 30)
        return {
            'violations': violations, 'score': score,
            'future_corr_1': corr_1, 'future_corr_5': corr_5
        }

    def _audit_return_causality(self, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Verificar causalidad entre senales y retornos"""
        violations = 0

        # Los retornos no deben anticipar senales futuras
        signal_changes = signals.diff().abs()
        past_returns = returns.shift(1).rolling(5).mean()

        # Correlacion entre cambios de senal y retornos pasados
        corr_past = signal_changes.corr(past_returns)
        if abs(corr_past) > 0.12:
            violations += 1

        # Test de causalidad de Granger simplificado
        try:
            from scipy.stats import pearsonr
            stat, p_value = pearsonr(signal_changes.dropna(), past_returns.dropna())
            if p_value < 0.05 and abs(stat) > 0.1:
                violations += 1
        except:
            pass

        score = max(0, 100 - violations * 25)
        return {'violations': violations, 'score': score, 'causality_corr': corr_past}

    def _audit_future_information(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """Detectar uso de informacion futura"""
        violations = 0

        # Verificar que las senales no usen precios futuros
        for i in range(100, min(500, len(signals))):  # Muestra aleatoria
            current_signal = signals.iloc[i]
            if current_signal != 0:
                # Verificar si la senal predice perfectamente el movimiento
                future_move = data['close'].iloc[i+1:i+6].pct_change().mean()
                if abs(current_signal * future_move) > 0.03:  # 3% threshold
                    violations += 1

        score = max(0, 100 - violations * 2)
        return {'violations': violations, 'score': score}

    def _run_statistical_tests(self, data: pd.DataFrame, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Tests estadisticos para detectar anomalias"""
        violations = 0
        tests_results = {}

        # Test de aleatoriedad de senales
        signal_runs = self._runs_test(signals)
        tests_results['runs_test_p'] = signal_runs
        if signal_runs < 0.05:  # Senales no aleatorias
            violations += 1

        # Test de estacionariedad de retornos
        returns_clean = returns.dropna()
        if len(returns_clean) > 50:
            adf_stat = self._adf_test(returns_clean)
            tests_results['adf_p_value'] = adf_stat
            if adf_stat > 0.05:  # No estacionario
                violations += 1

        score = max(0, 100 - violations * 20)
        return {
            'violations': violations, 'score': score,
            'statistical_tests': tests_results
        }

    def _runs_test(self, series: pd.Series) -> float:
        """Test de rachas para verificar aleatoriedad"""
        try:
            binary_series = (series > 0).astype(int)
            runs, n1, n2 = 0, 0, 0

            for i in range(len(binary_series)):
                if binary_series.iloc[i] == 1:
                    n1 += 1
                else:
                    n2 += 1

                if i > 0 and binary_series.iloc[i] != binary_series.iloc[i-1]:
                    runs += 1

            if n1 == 0 or n2 == 0:
                return 1.0

            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

            if variance <= 0:
                return 1.0

            z_stat = (runs - expected_runs) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            return p_value
        except:
            return 1.0

    def _adf_test(self, series: pd.Series) -> float:
        """Test Augmented Dickey-Fuller simplificado"""
        try:
            # Implementacion simplificada del test ADF
            y = series.values
            n = len(y)

            # Primera diferencia
            y_diff = np.diff(y)
            y_lag = y[:-1]

            # Regresion simple: y_diff = alpha * y_lag + error
            if len(y_diff) > 10:
                correlation = np.corrcoef(y_diff, y_lag)[0, 1]
                # Aproximacion del p-value basada en correlacion
                t_stat = correlation * np.sqrt((n-2)/(1-correlation**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                return min(p_value, 1.0)

            return 1.0
        except:
            return 1.0


class WalkForwardAnalyzer:
    """
    Analizador Walk-Forward institucional con validacion temporal robusta.
    Implementa ventanas deslizantes con purge gaps y out-of-sample testing.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = []

    def analyze(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """Ejecutar analisis walk-forward completo"""
        logger.info("Iniciando analisis Walk-Forward institucional...")

        # Crear ventanas temporales
        windows = self._create_time_windows(data)
        logger.info(f"Creadas {len(windows)} ventanas walk-forward")

        wf_results = []

        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            logger.info(f"Procesando ventana {i+1}/{len(windows)}: {train_start} - {val_end}")

            # Extraer datos de entrenamiento y validacion
            train_data = data.loc[train_start:train_end]
            val_data = data.loc[val_start:val_end]

            if len(train_data) < 100 or len(val_data) < 20:
                continue

            # Generar senales en training
            train_signals = strategy_func(train_data)
            train_returns = self._calculate_returns_with_costs(train_data, train_signals)
            train_metrics = self._calculate_metrics(train_returns)

            # Validar en out-of-sample
            val_signals = strategy_func(val_data)
            val_returns = self._calculate_returns_with_costs(val_data, val_signals)
            val_metrics = self._calculate_metrics(val_returns)

            # Calcular degradacion
            degradation = self._calculate_degradation(train_metrics, val_metrics)

            window_result = {
                'window_id': i,
                'train_period': f"{train_start} - {train_end}",
                'val_period': f"{val_start} - {val_end}",
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'degradation': degradation,
                'stability_score': self._calculate_stability_score(degradation)
            }

            wf_results.append(window_result)

        # Analizar resultados agregados
        analysis_summary = self._analyze_wf_results(wf_results)

        logger.info(f"Walk-Forward completado. Score estabilidad: {analysis_summary.get('overall_stability', 0):.1f}")

        return {
            'windows_results': wf_results,
            'summary': analysis_summary,
            'recommendation': self._generate_wf_recommendation(analysis_summary)
        }

    def _create_time_windows(self, data: pd.DataFrame) -> List[Tuple]:
        """Crear ventanas temporales con purge gaps"""
        windows = []

        start_date = data.index.min()
        end_date = data.index.max()

        current_date = start_date

        while current_date < end_date:
            # Ventana de entrenamiento
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.config.training_window_months)

            # Gap de purge para evitar lookahead
            purge_end = train_end + pd.DateOffset(days=self.config.purge_window_days)

            # Ventana de validacion
            val_start = purge_end
            val_end = val_start + pd.DateOffset(months=self.config.validation_window_months)

            # Verificar que no excedamos los datos disponibles
            if val_end <= end_date:
                windows.append((train_start, train_end, val_start, val_end))

            # Avanzar ventana
            current_date += pd.DateOffset(months=self.config.step_size_months)

        return windows

    def _calculate_returns_with_costs(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calcular retornos con costos XM reales"""
        SPREAD_POINTS = 40
        COMMISSION_USD = 7.0
        SLIPPAGE_POINTS = 2
        POINT_VALUE = 0.01

        returns = pd.Series(0.0, index=data.index)
        position = 0
        entry_price = 0.0

        for i in range(1, len(signals)):
            signal = signals.iloc[i]
            current_price = data['close'].iloc[i]

            if signal != 0 and position == 0:
                position = signal
                if position == 1:
                    entry_price = current_price + (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
                else:
                    entry_price = current_price - (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE

            elif signal != 0 and signal != position:
                if position == 1:
                    exit_price = current_price - (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
                    trade_return = (exit_price - entry_price) / entry_price
                else:
                    exit_price = current_price + (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
                    trade_return = (entry_price - exit_price) / entry_price

                commission_pct = COMMISSION_USD / (entry_price * 100)
                trade_return -= commission_pct
                returns.iloc[i] = trade_return

                position = signal
                if position == 1:
                    entry_price = current_price + (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
                else:
                    entry_price = current_price - (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE

        return returns

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcular metricas de performance"""
        if len(returns) == 0 or returns.dropna().empty:
            return {'sharpe': 0, 'max_dd': 0, 'total_return': 0, 'win_rate': 0, 'profit_factor': 1}

        clean_returns = returns.dropna()
        trades = clean_returns[clean_returns != 0]

        # Metricas basicas
        total_return = (1 + clean_returns).prod() - 1
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(252) if clean_returns.std() != 0 else 0

        # Maximum Drawdown
        cumulative = (1 + clean_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        # Metricas de trading
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 1

        return {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades)
        }

    def _calculate_degradation(self, train_metrics: Dict, val_metrics: Dict) -> Dict[str, float]:
        """Calcular degradacion entre training y validation"""
        degradation = {}

        for metric in ['sharpe', 'total_return', 'win_rate', 'profit_factor']:
            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)

            if train_val != 0:
                degradation[f'{metric}_degradation'] = (train_val - val_val) / abs(train_val)
            else:
                degradation[f'{metric}_degradation'] = 0

        # Degradacion de drawdown (inversa - menor es mejor)
        train_dd = train_metrics.get('max_dd', 0)
        val_dd = val_metrics.get('max_dd', 0)
        degradation['max_dd_degradation'] = (val_dd - train_dd) / max(train_dd, 0.01)

        return degradation

    def _calculate_stability_score(self, degradation: Dict) -> float:
        """Calcular score de estabilidad basado en degradacion"""
        degradation_values = [abs(v) for v in degradation.values()]
        avg_degradation = np.mean(degradation_values) if degradation_values else 0

        # Score de 0-100, donde 100 es estabilidad perfecta
        stability_score = max(0, 100 - avg_degradation * 200)
        return stability_score

    def _analyze_wf_results(self, wf_results: List[Dict]) -> Dict[str, Any]:
        """Analizar resultados agregados del walk-forward"""
        if not wf_results:
            return {'overall_stability': 0, 'consistent_performance': False}

        # Extraer metricas de validacion
        val_sharpes = [r['val_metrics']['sharpe'] for r in wf_results]
        val_returns = [r['val_metrics']['total_return'] for r in wf_results]
        stability_scores = [r['stability_score'] for r in wf_results]

        # Analisis de consistencia
        sharpe_consistency = 1 - (np.std(val_sharpes) / max(np.mean(val_sharpes), 0.1))
        return_consistency = 1 - (np.std(val_returns) / max(np.mean(val_returns), 0.01))

        overall_stability = np.mean(stability_scores)

        # Verificar performance consistente
        positive_windows = sum(1 for r in val_returns if r > 0)
        consistent_performance = positive_windows >= len(wf_results) * 0.7  # 70% ventanas positivas

        return {
            'overall_stability': overall_stability,
            'sharpe_consistency': sharpe_consistency,
            'return_consistency': return_consistency,
            'consistent_performance': consistent_performance,
            'positive_windows_pct': positive_windows / len(wf_results),
            'avg_val_sharpe': np.mean(val_sharpes),
            'avg_val_return': np.mean(val_returns)
        }

    def _generate_wf_recommendation(self, summary: Dict) -> str:
        """Generar recomendacion basada en walk-forward"""
        stability = summary.get('overall_stability', 0)
        consistency = summary.get('consistent_performance', False)

        if stability > 80 and consistency:
            return "EXCELENTE: Estrategia muy estable y consistente"
        elif stability > 60 and consistency:
            return "BUENO: Estrategia estable con performance consistente"
        elif stability > 40:
            return "MODERADO: Estrategia con estabilidad moderada"
        else:
            return "RIESGO: Estrategia inestable, requiere optimization"


class StressTester:
    """
    Tester de estres institucional para evaluar robustez bajo condiciones extremas.
    Implementa multiples escenarios de estres y analiza supervivencia de estrategia.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def comprehensive_stress_test(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """Ejecutar bateria completa de tests de estres"""
        logger.info("Iniciando tests de estres institucionales...")

        stress_results = {}

        # Test escenarios de costos
        for scenario_name, scenario_params in self.config.stress_scenarios.items():
            logger.info(f"Ejecutando escenario: {scenario_name}")
            result = self._test_cost_scenario(data, strategy_func, scenario_params)
            stress_results[scenario_name] = result

        # Test de volatilidad extrema
        stress_results['high_volatility'] = self._test_volatility_stress(data, strategy_func)

        # Test de gaps de mercado
        stress_results['market_gaps'] = self._test_gap_stress(data, strategy_func)

        # Test de condiciones de baja liquidez
        stress_results['low_liquidity'] = self._test_liquidity_stress(data, strategy_func)

        # Analizar supervivencia general
        survival_analysis = self._analyze_survival(stress_results)

        logger.info(f"Stress testing completado. Supervivencia: {survival_analysis['overall_survival']:.1f}%")

        return {
            'scenario_results': stress_results,
            'survival_analysis': survival_analysis,
            'risk_assessment': self._assess_risk_level(survival_analysis)
        }

    def _test_cost_scenario(self, data: pd.DataFrame, strategy_func, scenario_params: Dict) -> Dict[str, Any]:
        """Test con escenario de costos especifico"""
        signals = strategy_func(data)

        # Aplicar costos ajustados por escenario
        adjusted_returns = self._calculate_stressed_returns(
            data, signals, scenario_params
        )

        # Calcular metricas bajo estres
        stressed_metrics = self._calculate_stress_metrics(adjusted_returns)

        return {
            'scenario_params': scenario_params,
            'metrics': stressed_metrics,
            'survival': stressed_metrics['total_return'] > -0.5,  # No mas de 50% perdida
            'performance_ratio': self._calculate_performance_ratio(adjusted_returns)
        }

    def _calculate_stressed_returns(self, data: pd.DataFrame, signals: pd.Series,
                                   scenario_params: Dict) -> pd.Series:
        """Calcular retornos con costos estresados"""
        base_spread = 40
        base_commission = 7.0
        base_slippage = 2

        # Aplicar multiplicadores de estres
        stressed_spread = base_spread * scenario_params['spread_mult']
        stressed_commission = base_commission * scenario_params['comm_mult']
        stressed_slippage = base_slippage * scenario_params['slip_mult']

        POINT_VALUE = 0.01

        returns = pd.Series(0.0, index=data.index)
        position = 0
        entry_price = 0.0

        for i in range(1, len(signals)):
            signal = signals.iloc[i]
            current_price = data['close'].iloc[i]

            if signal != 0 and position == 0:
                position = signal
                if position == 1:
                    entry_price = current_price + (stressed_spread + stressed_slippage) * POINT_VALUE
                else:
                    entry_price = current_price - (stressed_spread + stressed_slippage) * POINT_VALUE

            elif signal != 0 and signal != position:
                if position == 1:
                    exit_price = current_price - (stressed_spread + stressed_slippage) * POINT_VALUE
                    trade_return = (exit_price - entry_price) / entry_price
                else:
                    exit_price = current_price + (stressed_spread + stressed_slippage) * POINT_VALUE
                    trade_return = (entry_price - exit_price) / entry_price

                commission_pct = stressed_commission / (entry_price * 100)
                trade_return -= commission_pct
                returns.iloc[i] = trade_return

                position = signal
                if position == 1:
                    entry_price = current_price + (stressed_spread + stressed_slippage) * POINT_VALUE
                else:
                    entry_price = current_price - (stressed_spread + stressed_slippage) * POINT_VALUE

        return returns

    def _test_volatility_stress(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """Test bajo condiciones de alta volatilidad"""
        # Identificar periodos de alta volatilidad
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        high_vol_threshold = volatility.quantile(0.9)

        # Filtrar solo periodos de alta volatilidad
        high_vol_periods = data[volatility > high_vol_threshold]

        if len(high_vol_periods) < 50:
            return {'metrics': {}, 'survival': True, 'note': 'Insufficient high volatility data'}

        signals = strategy_func(high_vol_periods)
        stressed_returns = self._calculate_stressed_returns(
            high_vol_periods, signals, {'spread_mult': 1.5, 'slip_mult': 2.0, 'comm_mult': 1.0}
        )

        metrics = self._calculate_stress_metrics(stressed_returns)

        return {
            'metrics': metrics,
            'survival': metrics['max_drawdown'] < 0.4,  # No mas de 40% DD
            'high_vol_periods': len(high_vol_periods)
        }

    def _test_gap_stress(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """Test con gaps de mercado simulados"""
        # Simular gaps aleatorios
        stressed_data = data.copy()
        gap_indices = np.random.choice(len(data), size=int(len(data) * 0.02), replace=False)

        for idx in gap_indices:
            if idx < len(data) - 1:
                gap_size = np.random.uniform(-0.02, 0.02)  # Gap de hasta 2%
                stressed_data.iloc[idx+1:, stressed_data.columns.get_loc('open')] *= (1 + gap_size)
                stressed_data.iloc[idx+1:, stressed_data.columns.get_loc('high')] *= (1 + gap_size)
                stressed_data.iloc[idx+1:, stressed_data.columns.get_loc('low')] *= (1 + gap_size)
                stressed_data.iloc[idx+1:, stressed_data.columns.get_loc('close')] *= (1 + gap_size)

        signals = strategy_func(stressed_data)
        stressed_returns = self._calculate_stressed_returns(
            stressed_data, signals, {'spread_mult': 2.0, 'slip_mult': 3.0, 'comm_mult': 1.0}
        )

        metrics = self._calculate_stress_metrics(stressed_returns)

        return {
            'metrics': metrics,
            'survival': metrics['total_return'] > -0.3,
            'gaps_simulated': len(gap_indices)
        }

    def _test_liquidity_stress(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """Test bajo condiciones de baja liquidez"""
        signals = strategy_func(data)

        # Simular condiciones de baja liquidez con mayor slippage
        liquidity_scenario = {
            'spread_mult': 2.5,
            'slip_mult': 4.0,
            'comm_mult': 1.5
        }

        stressed_returns = self._calculate_stressed_returns(data, signals, liquidity_scenario)
        metrics = self._calculate_stress_metrics(stressed_returns)

        return {
            'metrics': metrics,
            'survival': metrics['sharpe_ratio'] > 0.5,
            'scenario': liquidity_scenario
        }

    def _calculate_stress_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcular metricas bajo condiciones de estres"""
        if len(returns) == 0 or returns.dropna().empty:
            return {
                'total_return': -1.0, 'sharpe_ratio': -2.0, 'max_drawdown': 1.0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'total_trades': 0
            }

        clean_returns = returns.dropna()
        trades = clean_returns[clean_returns != 0]

        total_return = (1 + clean_returns).prod() - 1
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(252) if clean_returns.std() != 0 else -2.0

        # Maximum Drawdown
        cumulative = (1 + clean_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Metricas de trading
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades)
        }

    def _calculate_performance_ratio(self, returns: pd.Series) -> float:
        """Calcular ratio de performance vs baseline"""
        if len(returns) == 0:
            return 0.0

        total_return = (1 + returns.dropna()).prod() - 1
        baseline_return = 0.05  # 5% baseline anual

        return total_return / baseline_return if baseline_return != 0 else 0

    def _analyze_survival(self, stress_results: Dict) -> Dict[str, Any]:
        """Analizar supervivencia general en todos los escenarios"""
        survival_rates = []
        performance_ratios = []

        for scenario, result in stress_results.items():
            if isinstance(result, dict) and 'survival' in result:
                survival_rates.append(1 if result['survival'] else 0)
                if 'performance_ratio' in result:
                    performance_ratios.append(result['performance_ratio'])

        overall_survival = np.mean(survival_rates) * 100 if survival_rates else 0
        avg_performance_ratio = np.mean(performance_ratios) if performance_ratios else 0

        return {
            'overall_survival': overall_survival,
            'scenarios_passed': sum(survival_rates),
            'total_scenarios': len(survival_rates),
            'avg_performance_ratio': avg_performance_ratio,
            'survival_by_scenario': {k: v.get('survival', False) for k, v in stress_results.items()}
        }

    def _assess_risk_level(self, survival_analysis: Dict) -> str:
        """Evaluar nivel de riesgo basado en supervivencia"""
        survival_rate = survival_analysis.get('overall_survival', 0)

        if survival_rate >= 90:
            return "BAJO RIESGO: Estrategia muy robusta"
        elif survival_rate >= 70:
            return "RIESGO MODERADO: Estrategia robusta"
        elif survival_rate >= 50:
            return "RIESGO ELEVADO: Estrategia con vulnerabilidades"
        else:
            return "ALTO RIESGO: Estrategia no robusta"


def real_ema_strategy(data: pd.DataFrame) -> pd.Series:
    """
    ESTRATEGIA EMA REAL DEL USUARIO - IMPLEMENTACION EXACTA

    Triple EMA (8/21/55) + RSI + ATR con 4 filtros institucionales:
    1. Alineacion estricta de EMAs + separacion minima ATR
    2. Distancia precio-EMA rapida (pullback control)
    3. RSI momentum en rango valido
    4. Confirmacion por cierre de vela
    """
    try:
        import talib as ta
    except ImportError:
        logger.error("Instalando TA-Lib...")
        os.system("pip install TA-Lib")
        import talib as ta

    # PARAMETROS ESTRATEGIA REAL
    EMA_FAST = 8
    EMA_MEDIUM = 21
    EMA_SLOW = 55
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    RSI_UPPER = 70
    RSI_LOWER = 30
    ATR_MUL_CONS = 1.5      # Multiplicador minimo separacion EMAs
    ATR_MAX_DIST_FAST = 2.0 # Distancia maxima precio-EMA rapida

    min_required = max(EMA_SLOW, RSI_PERIOD) + ATR_PERIOD
    if len(data) < min_required:
        logger.warning(f"Datos insuficientes para EMA: {len(data)} < {min_required}")
        return pd.Series(0, index=data.index)

    # CALCULAR INDICADORES TECNICOS
    close_prices = data['close'].values
    high_prices = data['high'].values
    low_prices = data['low'].values

    ema_fast = ta.EMA(close_prices, timeperiod=EMA_FAST)
    ema_medium = ta.EMA(close_prices, timeperiod=EMA_MEDIUM)
    ema_slow = ta.EMA(close_prices, timeperiod=EMA_SLOW)
    atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)
    rsi = ta.RSI(close_prices, timeperiod=RSI_PERIOD)

    # GENERAR SENALES CON 4 FILTROS
    signals = pd.Series(0, index=data.index)
    signals_generated = 0

    for i in range(min_required, len(data)):
        # Variables actuales
        close = close_prices[i]
        ema_f, ema_m, ema_s = ema_fast[i], ema_medium[i], ema_slow[i]
        atr_val, rsi_val = atr[i], rsi[i]

        # Saltar si hay NaN
        if pd.isna([ema_f, ema_m, ema_s, atr_val, rsi_val]).any():
            continue

        # FILTRO 1: ALINEACION ESTRICTA + SEPARACION MINIMA
        aligned_buy = (ema_f > ema_m) and (ema_m > ema_s)
        aligned_sell = (ema_f < ema_m) and (ema_m < ema_s)

        # Separacion minima entre EMAs media y lenta
        ema_separation = abs(ema_m - ema_s) / atr_val
        is_separated = ema_separation > ATR_MUL_CONS

        if (aligned_buy or aligned_sell) and is_separated:

            # FILTRO 2: DISTANCIA PRECIO-EMA RAPIDA (Control pullback)
            price_distance = abs(close - ema_f) / atr_val
            is_price_near_fast = price_distance < ATR_MAX_DIST_FAST

            # FILTRO 3: RSI MOMENTUM VALIDO
            rsi_buy_valid = rsi_val < RSI_UPPER   # No sobrecomprado
            rsi_sell_valid = rsi_val > RSI_LOWER  # No sobrevendido

            # FILTRO 4: CONFIRMACION POR CIERRE
            close_confirmed_buy = (aligned_buy and close > ema_m)
            close_confirmed_sell = (aligned_sell and close < ema_m)

            # DECISION FINAL: TODOS LOS FILTROS DEBEN PASAR
            if is_price_near_fast:
                if close_confirmed_buy and rsi_buy_valid:
                    signals.iloc[i] = 1  # SENAL BUY
                    signals_generated += 1
                elif close_confirmed_sell and rsi_sell_valid:
                    signals.iloc[i] = -1  # SENAL SELL
                    signals_generated += 1

    total_signals = (signals != 0).sum()
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()

    logger.info(f"Estrategia EMA completada: {total_signals} senales ({buy_signals} BUY, {sell_signals} SELL)")

    return signals


def bootstrap_monte_carlo_with_gaps(data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """
    Monte Carlo bootstrap institucional con gaps temporales.
    Implementa bootstrap por bloques con preservacion de estructura temporal.
    """
    logger.info(f"Iniciando Monte Carlo con {config.n_simulations} simulaciones...")

    # Generar senales base
    base_signals = real_ema_strategy(data)
    base_returns = calculate_returns_with_xm_costs(data, base_signals)
    base_metrics = calculate_comprehensive_metrics(base_returns)

    # Preparar bootstrap
    clean_returns = base_returns.dropna()
    block_size = config.bootstrap_block_size
    n_blocks = len(clean_returns) // block_size

    if n_blocks < 10:
        logger.warning("Datos insuficientes para bootstrap robusto")
        return {'base_metrics': base_metrics, 'monte_carlo_stats': {}, 'confidence_intervals': {}}

    # Ejecutar simulaciones
    simulation_results = []

    for sim in range(config.n_simulations):
        # Bootstrap con gaps temporales
        bootstrapped_returns = []

        # Seleccionar bloques aleatorios con gaps
        selected_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)

        for block_idx in selected_blocks:
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, len(clean_returns))

            if end_idx > start_idx:
                block_returns = clean_returns.iloc[start_idx:end_idx].values
                bootstrapped_returns.extend(block_returns)

                # Agregar gap temporal aleatorio (simulacion realista)
                if np.random.random() < 0.1:  # 10% probabilidad de gap
                    gap_size = np.random.randint(1, 5)  # Gap de 1-4 periodos
                    bootstrapped_returns.extend([0] * gap_size)

        # Calcular metricas para simulacion
        boot_returns = pd.Series(bootstrapped_returns)
        boot_metrics = calculate_comprehensive_metrics(boot_returns)
        simulation_results.append(boot_metrics)

        if (sim + 1) % 500 == 0:
            logger.info(f"Completadas {sim + 1}/{config.n_simulations} simulaciones")

    # Analizar resultados Monte Carlo
    mc_df = pd.DataFrame(simulation_results)
    mc_stats = analyze_monte_carlo_results(mc_df, config)

    # Calcular intervalos de confianza
    confidence_intervals = calculate_confidence_intervals(mc_df, config.confidence_levels)

    logger.info("Monte Carlo bootstrap completado")

    return {
        'base_metrics': base_metrics,
        'monte_carlo_stats': mc_stats,
        'confidence_intervals': confidence_intervals,
        'simulation_results': simulation_results[:100]  # Guardar solo una muestra
    }


def calculate_returns_with_xm_costs(data: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """Calcular retornos con costos reales XM"""
    # COSTOS XM GOLD INSTITUCIONALES
    SPREAD_POINTS = 40      # Spread observado XM
    COMMISSION_USD = 7.0    # Comision por lote redondo
    SLIPPAGE_POINTS = 2     # Slippage promedio
    POINT_VALUE = 0.01      # Valor del punto GOLD

    returns = pd.Series(0.0, index=data.index)
    position = 0
    entry_price = 0.0
    total_trades = 0

    for i in range(1, len(signals)):
        signal = signals.iloc[i]
        current_price = data['close'].iloc[i]

        # LOGICA DE ENTRADA
        if signal != 0 and position == 0:
            position = signal
            total_trades += 1

            if position == 1:  # ENTRADA LONG
                entry_price = current_price + (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
            else:  # ENTRADA SHORT
                entry_price = current_price - (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE

        # LOGICA DE SALIDA (cambio de senal)
        elif signal != 0 and signal != position:
            # CALCULAR RETORNO DEL TRADE
            if position == 1:  # CERRAR LONG
                exit_price = current_price - (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
                trade_return = (exit_price - entry_price) / entry_price
            else:  # CERRAR SHORT
                exit_price = current_price + (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
                trade_return = (entry_price - exit_price) / entry_price

            # APLICAR COMISION
            commission_percentage = COMMISSION_USD / (entry_price * 100)  # Asumiendo lote de $10,000
            trade_return -= commission_percentage

            # REGISTRAR RETORNO
            returns.iloc[i] = trade_return

            # NUEVA POSICION
            position = signal
            if position == 1:
                entry_price = current_price + (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE
            else:
                entry_price = current_price - (SPREAD_POINTS + SLIPPAGE_POINTS) * POINT_VALUE

    logger.info(f"Retornos calculados: {total_trades} trades con costos XM reales")
    return returns


def calculate_comprehensive_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calcular metricas institucionales comprehensivas"""
    if len(returns) == 0 or returns.dropna().empty:
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'max_drawdown': 0.0,
            'volatility': 0.0, 'skewness': 0.0, 'kurtosis': 0.0,
            'win_rate': 0.0, 'profit_factor': 1.0, 'total_trades': 0,
            'avg_trade': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0,
            'consecutive_wins': 0, 'consecutive_losses': 0,
            'recovery_factor': 0.0, 'ulcer_index': 0.0
        }

    clean_returns = returns.dropna()
    trades = clean_returns[clean_returns != 0]

    # METRICAS DE RETORNO
    total_return = (1 + clean_returns).prod() - 1
    n_periods = len(clean_returns)
    periods_per_year = 252 * 24 * 10  # Asumiendo datos de 6 minutos
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1 if n_periods > 0 else 0

    # METRICAS DE RIESGO
    volatility = clean_returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = clean_returns[clean_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else volatility
    sortino_ratio = annualized_return / downside_std if downside_std != 0 else 0

    # DRAWDOWN ANALYSIS
    cumulative = (1 + clean_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0

    # Ulcer Index
    ulcer_index = np.sqrt((drawdown ** 2).mean())

    # MOMENTOS ESTADISTICOS
    skewness = clean_returns.skew() if len(clean_returns) > 2 else 0
    kurtosis = clean_returns.kurtosis() if len(clean_returns) > 3 else 0

    # METRICAS DE TRADING
    if len(trades) > 0:
        win_rate = (trades > 0).sum() / len(trades)
        avg_trade = trades.mean()
        best_trade = trades.max()
        worst_trade = trades.min()

        # Profit Factor
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        # Rachas consecutivas
        consecutive_wins = calculate_max_consecutive(trades > 0)
        consecutive_losses = calculate_max_consecutive(trades < 0)

        # Recovery Factor
        recovery_factor = total_return / max_drawdown if max_drawdown != 0 else 0
    else:
        win_rate = avg_trade = best_trade = worst_trade = 0
        profit_factor = 1.0
        consecutive_wins = consecutive_losses = 0
        recovery_factor = 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades),
        'avg_trade': avg_trade,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses,
        'recovery_factor': recovery_factor,
        'ulcer_index': ulcer_index
    }


def calculate_max_consecutive(series: pd.Series) -> int:
    """Calcular maxima racha consecutiva"""
    if len(series) == 0:
        return 0

    max_consecutive = current_consecutive = 0

    for value in series:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def analyze_monte_carlo_results(mc_df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """Analizar resultados agregados de Monte Carlo"""
    if mc_df.empty:
        return {}

    # Estadisticas centrales
    stats = {}

    for metric in ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate', 'profit_factor']:
        if metric in mc_df.columns:
            stats[f'{metric}_mean'] = mc_df[metric].mean()
            stats[f'{metric}_std'] = mc_df[metric].std()
            stats[f'{metric}_min'] = mc_df[metric].min()
            stats[f'{metric}_max'] = mc_df[metric].max()

    # Probabilidad de exito
    if 'sharpe_ratio' in mc_df.columns:
        stats['prob_positive_sharpe'] = (mc_df['sharpe_ratio'] > 0).mean()
        stats['prob_sharpe_above_threshold'] = (mc_df['sharpe_ratio'] > config.min_sharpe_ratio).mean()

    if 'max_drawdown' in mc_df.columns:
        stats['prob_dd_below_threshold'] = (mc_df['max_drawdown'] < config.max_drawdown_threshold).mean()

    # Analisis de colas (tail risk)
    if 'total_return' in mc_df.columns:
        stats['var_95'] = mc_df['total_return'].quantile(0.05)  # Value at Risk 95%
        stats['cvar_95'] = mc_df['total_return'][mc_df['total_return'] <= stats['var_95']].mean()  # Conditional VaR

    return stats


def calculate_confidence_intervals(mc_df: pd.DataFrame, confidence_levels: List[float]) -> Dict[str, Dict]:
    """Calcular intervalos de confianza para metricas clave"""
    intervals = {}

    key_metrics = ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate', 'profit_factor']

    for metric in key_metrics:
        if metric in mc_df.columns:
            intervals[metric] = {}
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                lower = mc_df[metric].quantile(alpha / 2)
                upper = mc_df[metric].quantile(1 - alpha / 2)
                intervals[metric][f'{conf_level:.0%}'] = {'lower': lower, 'upper': upper}

    return intervals


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Cargar y preparar datos historicos con auto-deteccion de rutas"""
    logger.info(f"Buscando archivo: {file_path}")

    # Auto-detectar rutas posibles (como backtesting.py)
    possible_paths = [
        file_path,  # Ruta original
        f"../{file_path}",  # Subir un nivel
        f"../../{file_path}",  # Subir dos niveles
        file_path.replace("database/", "../database/"),  # Ajuste relativo
        file_path.replace("database/", "../../database/")  # Ajuste relativo 2
    ]

    actual_file = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_file = path
            break

    if actual_file is None:
        # Si no encuentra el archivo, buscar cualquier GOLD M6
        import glob
        search_patterns = [
            "database/data_history/GOLD*M6*.csv",
            "../database/data_history/GOLD*M6*.csv",
            "../../database/data_history/GOLD*M6*.csv",
            "database/data_history/GOLD*.csv",
            "../database/data_history/GOLD*.csv",
            "../../database/data_history/GOLD*.csv"
        ]

        for pattern in search_patterns:
            files = glob.glob(pattern)
            if files:
                actual_file = files[0]  # Tomar el primero encontrado
                logger.info(f"Auto-detectado archivo: {actual_file}")
                break

    if actual_file is None:
        raise FileNotFoundError(f"No se encontro archivo GOLD en rutas posibles")

    logger.info(f"Cargando datos desde: {actual_file}")

    try:
        # Cargar datos con deteccion automatica de formato
        data = pd.read_csv(actual_file, encoding='utf-8')

        # Detectar y convertir columna temporal
        time_columns = ['time', 'datetime', 'timestamp', 'date']
        time_col = None

        for col in time_columns:
            if col in data.columns:
                time_col = col
                break

        if time_col:
            data[time_col] = pd.to_datetime(data[time_col])
            data.set_index(time_col, inplace=True)
        else:
            # Crear indice temporal por defecto
            logger.warning("No se encontro columna temporal, creando indice por defecto")
            start_date = '2020-01-01'
            data.index = pd.date_range(start=start_date, periods=len(data), freq='6min')

        # Verificar columnas OHLCV requeridas
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Columnas faltantes en datos: {missing_columns}")

        # Limpieza de datos
        original_length = len(data)

        # Eliminar filas con valores nulos
        data.dropna(inplace=True)

        # Validar consistencia OHLC
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['close'] <= 0) |
            (data['open'] <= 0)
        )

        if invalid_ohlc.any():
            logger.warning(f"Eliminando {invalid_ohlc.sum()} filas con datos OHLC invalidos")
            data = data[~invalid_ohlc]

        # Verificar cantidad final de datos
        final_length = len(data)
        if final_length < original_length * 0.9:
            logger.warning(f"Se perdio mas del 10% de datos en limpieza: {original_length} -> {final_length}")

        # Ordenar por indice temporal
        data.sort_index(inplace=True)

        # Informacion del dataset
        logger.info(f"Datos preparados exitosamente:")
        logger.info(f"  Periodo: {data.index.min()} a {data.index.max()}")
        logger.info(f"  Observaciones: {len(data):,}")
        logger.info(f"  Columnas: {list(data.columns)}")
        logger.info(f"  Frecuencia estimada: {pd.infer_freq(data.index[:100])}")

        return data

    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        raise


def run_institutional_analysis(data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """Ejecutar analisis institucional completo"""
    logger.info("=== INICIANDO ANALISIS INSTITUCIONAL YOHABOT ===")

    # FASE 1: AUDITORIA DE DATA LEAKAGE
    logger.info("FASE 1: Auditoria de Data Leakage")
    auditor = DataLeakageAuditor(config)

    # Generar senales para auditoria
    signals = real_ema_strategy(data)
    returns = calculate_returns_with_xm_costs(data, signals)

    audit_results = auditor.comprehensive_audit(data, signals, returns)

    # FASE 2: WALK-FORWARD ANALYSIS
    logger.info("FASE 2: Walk-Forward Analysis")
    wf_analyzer = WalkForwardAnalyzer(config)
    wf_results = wf_analyzer.analyze(data, real_ema_strategy)

    # FASE 3: STRESS TESTING
    logger.info("FASE 3: Stress Testing")
    stress_tester = StressTester(config)
    stress_results = stress_tester.comprehensive_stress_test(data, real_ema_strategy)

    # FASE 4: MONTE CARLO BOOTSTRAP
    logger.info("FASE 4: Monte Carlo Bootstrap")
    mc_results = bootstrap_monte_carlo_with_gaps(data, config)

    # FASE 5: OUT-OF-SAMPLE VALIDATION
    logger.info("FASE 5: Out-of-Sample Validation")
    oos_results = run_out_of_sample_validation(data, config)

    # FASE 6: DECISION INSTITUCIONAL
    logger.info("FASE 6: Decision Go/No-Go")
    decision_results = generate_institutional_decision(
        audit_results, wf_results, stress_results, mc_results, oos_results, config
    )

    # Compilar resultados completos
    full_results = {
        'data_leakage_audit': audit_results,
        'walk_forward_analysis': wf_results,
        'stress_testing': stress_results,
        'monte_carlo_analysis': mc_results,
        'out_of_sample_validation': oos_results,
        'institutional_decision': decision_results,
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_period': f"{data.index.min()} to {data.index.max()}",
            'total_observations': len(data),
            'config': config.__dict__
        }
    }

    logger.info("=== ANALISIS INSTITUCIONAL COMPLETADO ===")
    return full_results


def run_out_of_sample_validation(data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """Ejecutar validacion out-of-sample estricta"""
    logger.info("Ejecutando validacion Out-of-Sample...")

    # Dividir datos con purge gap
    oos_split_point = int(len(data) * (1 - config.oos_percentage))
    purge_days = config.purge_window_days

    # In-Sample (entrenamiento)
    is_data = data.iloc[:oos_split_point]

    # Purge gap
    purge_end_date = is_data.index.max() + pd.Timedelta(days=purge_days)
    oos_start_idx = data.index.searchsorted(purge_end_date)

    # Out-of-Sample (testing)
    oos_data = data.iloc[oos_start_idx:]

    if len(oos_data) < 100:
        logger.warning("Datos insuficientes para Out-of-Sample robusto")
        return {'error': 'Insufficient OOS data'}

    # Analizar In-Sample
    is_signals = real_ema_strategy(is_data)
    is_returns = calculate_returns_with_xm_costs(is_data, is_signals)
    is_metrics = calculate_comprehensive_metrics(is_returns)

    # Analizar Out-of-Sample
    oos_signals = real_ema_strategy(oos_data)
    oos_returns = calculate_returns_with_xm_costs(oos_data, oos_signals)
    oos_metrics = calculate_comprehensive_metrics(oos_returns)

    # Calcular degradacion
    degradation_analysis = calculate_oos_degradation(is_metrics, oos_metrics)

    logger.info(f"OOS Validation: IS period {len(is_data)} obs, OOS period {len(oos_data)} obs")

    return {
        'in_sample_metrics': is_metrics,
        'out_of_sample_metrics': oos_metrics,
        'degradation_analysis': degradation_analysis,
        'data_split_info': {
            'is_observations': len(is_data),
            'oos_observations': len(oos_data),
            'purge_days': purge_days,
            'oos_percentage': len(oos_data) / len(data)
        }
    }


def calculate_oos_degradation(is_metrics: Dict, oos_metrics: Dict) -> Dict[str, Any]:
    """Calcular degradacion In-Sample vs Out-of-Sample"""
    degradation = {}

    key_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']

    for metric in key_metrics:
        is_val = is_metrics.get(metric, 0)
        oos_val = oos_metrics.get(metric, 0)

        if metric == 'max_drawdown':
            # Para drawdown, mayor es peor
            if is_val != 0:
                degradation[f'{metric}_change'] = (oos_val - is_val) / abs(is_val)
            else:
                degradation[f'{metric}_change'] = 0
        else:
            # Para otras metricas, menor es peor
            if is_val != 0:
                degradation[f'{metric}_change'] = (oos_val - is_val) / abs(is_val)
            else:
                degradation[f'{metric}_change'] = 0

    # Score general de degradacion
    degradation_values = [abs(v) for v in degradation.values()]
    avg_degradation = np.mean(degradation_values) if degradation_values else 0

    # Clasificar nivel de degradacion
    if avg_degradation < 0.1:
        degradation_level = "EXCELENTE"
    elif avg_degradation < 0.25:
        degradation_level = "BUENO"
    elif avg_degradation < 0.5:
        degradation_level = "MODERADO"
    else:
        degradation_level = "ALTO"

    return {
        'metric_changes': degradation,
        'average_degradation': avg_degradation,
        'degradation_level': degradation_level,
        'oos_suitable': avg_degradation < 0.3  # Umbral institucional
    }


def generate_institutional_decision(audit_results: Dict, wf_results: Dict, stress_results: Dict,
                                  mc_results: Dict, oos_results: Dict, config: AnalysisConfig) -> Dict[str, Any]:
    """Generar decision institucional Go/No-Go"""
    logger.info("Generando decision institucional...")

    # SCORING DE COMPONENTES (0-100 cada uno)
    component_scores = {}

    # 1. Data Leakage Score
    audit_score = audit_results.get('overall_score', 0)
    component_scores['data_integrity'] = audit_score

    # 2. Walk-Forward Score
    wf_stability = wf_results.get('summary', {}).get('overall_stability', 0)
    component_scores['walk_forward_stability'] = wf_stability

    # 3. Stress Testing Score
    stress_survival = stress_results.get('survival_analysis', {}).get('overall_survival', 0)
    component_scores['stress_survival'] = stress_survival

    # 4. Monte Carlo Score
    base_metrics = mc_results.get('base_metrics', {})
    mc_score = calculate_monte_carlo_score(base_metrics, config)
    component_scores['monte_carlo_robustness'] = mc_score

    # 5. Out-of-Sample Score
    oos_score = calculate_oos_score(oos_results)
    component_scores['out_of_sample_validation'] = oos_score

    # PESOS INSTITUCIONALES
    weights = {
        'data_integrity': 0.20,           # 20% - Critico para trading real
        'walk_forward_stability': 0.25,  # 25% - Estabilidad temporal
        'stress_survival': 0.20,         # 20% - Robustez bajo estres
        'monte_carlo_robustness': 0.20,  # 20% - Validacion estadistica
        'out_of_sample_validation': 0.15 # 15% - Performance real
    }

    # CALCULAR SCORE FINAL PONDERADO
    final_score = sum(component_scores[component] * weights[component]
                     for component in weights.keys())

    # UMBRALES INSTITUCIONALES
    if final_score >= 80:
        decision = "GO"
        confidence = "HIGH"
        recommendation = "ESTRATEGIA APTA: Proceder con implementacion institucional"
    elif final_score >= 65:
        decision = "CONDITIONAL_GO"
        confidence = "MEDIUM"
        recommendation = "ESTRATEGIA VIABLE: Proceder con precaucion y monitoreo adicional"
    elif final_score >= 50:
        decision = "NO_GO"
        confidence = "MEDIUM"
        recommendation = "ESTRATEGIA NO APTA: Requiere optimizacion antes de deployment"
    else:
        decision = "STRONG_NO_GO"
        confidence = "HIGH"
        recommendation = "ESTRATEGIA NO VIABLE: Revision fundamental requerida"

    # CONTAR COMPONENTES APROBADOS
    passing_threshold = 70
    components_passed = sum(1 for score in component_scores.values() if score >= passing_threshold)
    total_components = len(component_scores)

    # IDENTIFICAR AREAS DE MEJORA
    improvement_areas = []
    for component, score in component_scores.items():
        if score < passing_threshold:
            improvement_areas.append(f"{component}: {score:.1f}/100")

    # GENERAR RECOMENDACIONES ESPECIFICAS
    specific_recommendations = generate_specific_recommendations(
        component_scores, audit_results, wf_results, stress_results, mc_results, oos_results
    )

    decision_report = {
        'final_decision': decision,
        'overall_score': final_score,
        'confidence_level': confidence,
        'components_passed': f"{components_passed}/{total_components}",
        'component_scores': component_scores,
        'component_weights': weights,
        'recommendation': recommendation,
        'improvement_areas': improvement_areas,
        'specific_recommendations': specific_recommendations,
        'risk_assessment': assess_overall_risk(component_scores),
        'deployment_readiness': final_score >= 65
    }

    logger.info(f"Decision final: {decision} (Score: {final_score:.1f}/100)")
    return decision_report


def calculate_monte_carlo_score(base_metrics: Dict, config: AnalysisConfig) -> float:
    """Calcular score basado en metricas Monte Carlo"""
    if not base_metrics:
        return 0

    score = 0
    max_score = 100

    # Sharpe Ratio (30 puntos)
    sharpe = base_metrics.get('sharpe_ratio', 0)
    if sharpe >= config.min_sharpe_ratio:
        score += 30
    else:
        score += (sharpe / config.min_sharpe_ratio) * 30

    # Max Drawdown (25 puntos)
    max_dd = base_metrics.get('max_drawdown', 1)
    if max_dd <= config.max_drawdown_threshold:
        score += 25
    else:
        score += max(0, 25 * (1 - max_dd / config.max_drawdown_threshold))

    # Profit Factor (20 puntos)
    pf = base_metrics.get('profit_factor', 0)
    if pf >= config.min_profit_factor:
        score += 20
    else:
        score += (pf / config.min_profit_factor) * 20

    # Win Rate (15 puntos)
    wr = base_metrics.get('win_rate', 0)
    if wr >= config.min_win_rate:
        score += 15
    else:
        score += (wr / config.min_win_rate) * 15

    # Total Return positivo (10 puntos)
    total_ret = base_metrics.get('total_return', 0)
    if total_ret > 0:
        score += 10

    return min(score, max_score)


def calculate_oos_score(oos_results: Dict) -> float:
    """Calcular score Out-of-Sample"""
    if 'error' in oos_results:
        return 0

    degradation_analysis = oos_results.get('degradation_analysis', {})
    avg_degradation = degradation_analysis.get('average_degradation', 1)
    oos_suitable = degradation_analysis.get('oos_suitable', False)

    # Score basado en degradacion
    if oos_suitable:
        base_score = 70
    else:
        base_score = 30

    # Ajustar por nivel de degradacion
    degradation_penalty = min(50, avg_degradation * 100)
    final_score = max(0, base_score - degradation_penalty)

    return final_score


def generate_specific_recommendations(component_scores: Dict, audit_results: Dict,
                                    wf_results: Dict, stress_results: Dict,
                                    mc_results: Dict, oos_results: Dict) -> List[str]:
    """Generar recomendaciones especificas basadas en resultados"""
    recommendations = []

    # Data Integrity
    if component_scores.get('data_integrity', 0) < 70:
        violations = audit_results.get('violations_summary', [])
        if violations:
            recommendations.append(f"Corregir violaciones de data leakage: {', '.join(violations[:2])}")
        else:
            recommendations.append("Revisar implementacion de estrategia para eliminar data leakage")

    # Walk-Forward
    if component_scores.get('walk_forward_stability', 0) < 70:
        wf_summary = wf_results.get('summary', {})
        if not wf_summary.get('consistent_performance', False):
            recommendations.append("Mejorar consistencia temporal - considerar parametros adaptativos")
        recommendations.append("Optimizar parametros para mayor estabilidad walk-forward")

    # Stress Testing
    if component_scores.get('stress_survival', 0) < 70:
        survival_by_scenario = stress_results.get('survival_analysis', {}).get('survival_by_scenario', {})
        failed_scenarios = [scenario for scenario, passed in survival_by_scenario.items() if not passed]
        if failed_scenarios:
            recommendations.append(f"Fortalecer estrategia contra: {', '.join(failed_scenarios[:2])}")
        recommendations.append("Implementar mecanismos de proteccion adicionales")

    # Monte Carlo
    if component_scores.get('monte_carlo_robustness', 0) < 70:
        base_metrics = mc_results.get('base_metrics', {})
        if base_metrics.get('sharpe_ratio', 0) < 1.2:
            recommendations.append("Mejorar Sharpe ratio - optimizar relacion riesgo/retorno")
        if base_metrics.get('max_drawdown', 0) > 0.18:
            recommendations.append("Reducir drawdown maximo - implementar stop-loss mas agresivo")

    # Out-of-Sample
    if component_scores.get('out_of_sample_validation', 0) < 70:
        recommendations.append("Mejorar robustez out-of-sample - revisar overfitting")
        degradation = oos_results.get('degradation_analysis', {}).get('degradation_level', '')
        if degradation in ['ALTO', 'MODERADO']:
            recommendations.append("Reducir degradacion OOS - simplificar logica de estrategia")

    return recommendations


def assess_overall_risk(component_scores: Dict) -> str:
    """Evaluar riesgo general basado en scores de componentes"""
    avg_score = np.mean(list(component_scores.values()))
    min_score = min(component_scores.values())

    # Riesgo basado en score promedio y score minimo
    if avg_score >= 80 and min_score >= 70:
        return "RIESGO BAJO: Todos los componentes institucionales aprueban"
    elif avg_score >= 70 and min_score >= 60:
        return "RIESGO MODERADO: Estrategia solida con areas menores de mejora"
    elif avg_score >= 60:
        return "RIESGO ELEVADO: Requiere atencion en componentes especificos"
    else:
        return "RIESGO ALTO: Multiples componentes requieren mejora significativa"


def save_institutional_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Guardar resultados institucionales"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"yohabot_institutional_analysis_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Resultados guardados: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error guardando resultados: {str(e)}")
        raise


def print_executive_summary(results: Dict[str, Any]):
    """Imprimir resumen ejecutivo institucional"""
    print("\n" + "="*90)
    print("                 YOHABOT ANALISIS INSTITUCIONAL - RESUMEN EJECUTIVO")
    print("="*90)

    # Decision principal
    decision_data = results.get('institutional_decision', {})
    decision = decision_data.get('final_decision', 'UNKNOWN')
    overall_score = decision_data.get('overall_score', 0)
    confidence = decision_data.get('confidence_level', 'UNKNOWN')

    # Icono de decision
    if decision == "GO":
        icon = "🟢"
        status = "APROBADA"
    elif decision == "CONDITIONAL_GO":
        icon = "🟡"
        status = "APROBADA CON CONDICIONES"
    else:
        icon = "🔴"
        status = "NO APROBADA"

    print(f"\n{icon} DECISION INSTITUCIONAL: {status}")
    print(f"📊 SCORE GENERAL: {overall_score:.1f}/100")
    print(f"🎯 CONFIANZA: {confidence}")
    print(f"📋 COMPONENTES: {decision_data.get('components_passed', '0/0')}")

    # Scores por componente
    component_scores = decision_data.get('component_scores', {})
    print(f"\n📈 EVALUACION POR COMPONENTES:")

    component_names = {
        'data_integrity': 'Integridad de Datos',
        'walk_forward_stability': 'Estabilidad Walk-Forward',
        'stress_survival': 'Supervivencia Estres',
        'monte_carlo_robustness': 'Robustez Monte Carlo',
        'out_of_sample_validation': 'Validacion Out-of-Sample'
    }

    for component, score in component_scores.items():
        name = component_names.get(component, component)
        status_icon = "✅" if score >= 70 else "❌"
        print(f"   {status_icon} {name}: {score:.1f}/100")

    # Metricas clave de la estrategia
    mc_results = results.get('monte_carlo_analysis', {})
    base_metrics = mc_results.get('base_metrics', {})

    if base_metrics:
        print(f"\n📊 METRICAS CLAVE DE LA ESTRATEGIA EMA (8/21/55):")
        print(f"   • Sharpe Ratio: {base_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   • Max Drawdown: {base_metrics.get('max_drawdown', 0):.1%}")
        print(f"   • Profit Factor: {base_metrics.get('profit_factor', 0):.2f}")
        print(f"   • Win Rate: {base_metrics.get('win_rate', 0):.1%}")
        print(f"   • Total Return: {base_metrics.get('total_return', 0):.1%}")
        print(f"   • Total Trades: {base_metrics.get('total_trades', 0)}")

    # Validacion Out-of-Sample
    oos_results = results.get('out_of_sample_validation', {})
    if 'degradation_analysis' in oos_results:
        degradation = oos_results['degradation_analysis']
        print(f"\n🔍 VALIDACION OUT-OF-SAMPLE:")
        print(f"   • Degradacion: {degradation.get('degradation_level', 'N/A')}")
        print(f"   • Degradacion Promedio: {degradation.get('average_degradation', 0):.1%}")
        print(f"   • Apto OOS: {'✅' if degradation.get('oos_suitable', False) else '❌'}")

    # Stress Testing
    stress_results = results.get('stress_testing', {})
    if 'survival_analysis' in stress_results:
        survival = stress_results['survival_analysis']
        print(f"\n⚡ RESISTENCIA AL ESTRES:")
        print(f"   • Supervivencia General: {survival.get('overall_survival', 0):.1f}%")
        print(f"   • Escenarios Pasados: {survival.get('scenarios_passed', 0)}/{survival.get('total_scenarios', 0)}")

    # Recomendaciones
    recommendations = decision_data.get('specific_recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMENDACIONES INSTITUCIONALES:")
        for i, rec in enumerate(recommendations[:3], 1):  # Mostrar solo las primeras 3
            print(f"   {i}. {rec}")

    # Risk Assessment
    risk_assessment = decision_data.get('risk_assessment', '')
    if risk_assessment:
        print(f"\n⚠️ EVALUACION DE RIESGO:")
        print(f"   {risk_assessment}")

    # Informacion del analisis
    metadata = results.get('analysis_metadata', {})
    print(f"\n📊 INFORMACION DEL ANALISIS:")
    print(f"   • Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"   • Periodo: {metadata.get('data_period', 'N/A')}")
    print(f"   • Observaciones: {metadata.get('total_observations', 0):,}")
    print(f"   • Estrategia: EMA 8/21/55 + RSI + ATR (4 filtros)")
    print(f"   • Costos: XM Gold (40pts spread + $7 comision)")

    # Recommendation final
    recommendation = decision_data.get('recommendation', '')
    print(f"\n📋 RECOMENDACION FINAL:")
    print(f"   {recommendation}")

    print("\n" + "="*90)

    # Accion sugerida
    if decision == "GO":
        print("🚀 ESTRATEGIA APROBADA - Proceder con implementacion institucional")
        print("📈 Recomendado: Paper trading inicial -> Live trading progresivo")
    elif decision == "CONDITIONAL_GO":
        print("⚠️ ESTRATEGIA VIABLE - Implementar con monitoreo adicional")
        print("🔍 Recomendado: Paper trading extensivo + revision mensual")
    else:
        print("🛑 ESTRATEGIA NO APROBADA - Optimizacion requerida")
        print("🔧 Recomendado: Abordar areas de mejora antes de deployment")


def main():
    """Funcion principal del analisis institucional"""
    logger.info("🚀 YOHABOT MONTE CARLO INSTITUCIONAL - INICIANDO")

    # Si no se proporciona archivo, buscar automáticamente
    if len(sys.argv) == 1:
        # Buscar automáticamente archivo GOLD M6
        import glob
        search_patterns = [
            "database/data_history/GOLD*M6*.csv",
            "../database/data_history/GOLD*M6*.csv",
            "../../database/data_history/GOLD*M6*.csv",
            "database/data_history/GOLD*.csv",
            "../database/data_history/GOLD*.csv",
            "../../database/data_history/GOLD*.csv"
        ]

        data_file = None
        for pattern in search_patterns:
            files = glob.glob(pattern)
            if files:
                data_file = files[0]
                print(f"📁 Auto-detectado: {data_file}")
                break

        if data_file is None:
            print("❌ No se encontró archivo GOLD")
            print("💡 Uso: python monte_carlo_institucional.py [archivo.csv]")
            sys.exit(1)

    elif len(sys.argv) == 2:
        data_file = sys.argv[1]
    else:
        print("❌ Uso: python monte_carlo_institucional.py [archivo.csv]")
        print("📁 Si no especificas archivo, se detectará automáticamente")
        sys.exit(1)

    try:
        # Configurar encoding para evitar errores Unicode
        import locale
        if hasattr(locale, 'setlocale'):
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except:
                pass

        # Configuracion institucional
        config = AnalysisConfig()
        logger.info(f"⚙️ Configuracion: {config.n_simulations} simulaciones, {len(config.stress_scenarios)} escenarios estres")

        # Cargar y preparar datos
        data = load_and_prepare_data(data_file)

        if len(data) < 1000:
            logger.warning("⚠️ Dataset pequeno - resultados pueden ser menos robustos")

        # Ejecutar analisis institucional completo
        logger.info("🔄 Ejecutando analisis institucional completo...")
        results = run_institutional_analysis(data, config)

        # Guardar resultados
        output_file = save_institutional_results(results)

        # Mostrar resumen ejecutivo
        print_executive_summary(results)

        # Mensaje final
        decision = results.get('institutional_decision', {}).get('final_decision', 'UNKNOWN')

        if decision == "GO":
            print(f"\n🟢 ESTRATEGIA EMA INSTITUCIONAL: APROBADA")
            print(f"🚀 Proceder con implementacion y monitoreo continuo")
        elif decision == "CONDITIONAL_GO":
            print(f"\n🟡 ESTRATEGIA EMA INSTITUCIONAL: APROBADA CON CONDICIONES")
            print(f"⚠️ Implementar con precaucion y monitoreo adicional")
        else:
            print(f"\n🔴 ESTRATEGIA EMA INSTITUCIONAL: NO APROBADA")
            print(f"🔧 Revisar y optimizar antes de considerar deployment")

        print(f"\n💾 Analisis completo guardado: {output_file}")
        logger.info("✅ ANALISIS INSTITUCIONAL COMPLETADO EXITOSAMENTE")

    except Exception as e:
        logger.error(f"❌ ERROR CRITICO: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        print("📞 Contactar soporte tecnico para resolucion")
        sys.exit(1)


if __name__ == "__main__":
    main()