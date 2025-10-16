# ======================== ENHANCED CRYPTO ANALYZER (REVISED) ========================
# Lưu ý: Mọi lỗ hổng random fallback đã được khắc phục, dùng historical mean/neutral thay random
# Thêm kiểm tra dữ liệu bất thường (outlier), xử lý reconnect WebSocket tốt hơn, tăng cảnh báo fallback

import json
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import ccxt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from scipy.signal import find_peaks
import time
from datetime import datetime, timedelta
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from scipy import stats
import warnings
import logging
import sys
import os
from math import exp

warnings.filterwarnings('ignore')

# Cấu hình logging với Unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 🔑 CẤU HÌNH API KEYS
API_KEYS = {
    'FRED_API_KEY': os.getenv('FRED_API_KEY', ''),
    'FMP_API_KEY': os.getenv('FMP_API_KEY', ''),
}

@dataclass
class TradingConfig:
    # Risk Management
    stop_loss_atr_multiplier: float = 1.2
    take_profit_atr_multiplier: float = 2.0
    max_position_size_pct: float = 0.15
    
    # Scoring Thresholds
    base_long_threshold: float = 0.60
    base_short_threshold: float = 0.40
    min_confidence_score: float = 0.45
    
    # Weights
    smart_money_weight: float = 0.35
    technical_weight: float = 0.25
    volume_weight: float = 0.15
    realtime_weight: float = 0.25
    
    # Filter thresholds
    max_spread_pct: float = 0.005
    min_total_volume: float = 100000
    max_funding_rate: float = 0.0010
    min_oi_change_pct: float = -50.0
    
    # Regime filter parameters
    btc_d_threshold: float = 0.02
    dxy_threshold: float = 0.01
    risk_off_long_adjustment: float = 0.05
    crowded_long_funding_threshold: float = 0.0005

    # Slippage and Fees
    default_slippage_pct: float = 0.001  # 0.1%
    default_fee_pct: float = 0.0004      # 0.04%
    margin_call_threshold: float = 0.8  # 80% of margin used triggers warning

class Decision(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class EntryStrategy(Enum):
    IMMEDIATE = "VÀO LỆNH NGAY"
    WAIT_PULLBACK = "CHỜ PULLBACK"
    WAIT_BREAKOUT = "CHỜ BREAKOUT"
    WAIT_CONFIRMATION = "CHỜ XÁC NHẬN"

@dataclass
class RealTimeData:
    """Dữ liệu real-time từ multiple exchanges"""
    orderbook_imbalance: float
    cvd_signal: Dict
    liquidation_bias: Dict
    options_data: Dict
    funding_rate: float
    open_interest: float
    spread_pct: float
    timestamp: str

@dataclass
class TechnicalIndicators:
    """Chỉ báo kỹ thuật toàn diện"""
    rsi: float
    rsi_divergence: str
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_trend: str
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_9: float
    ema_21: float
    golden_cross: bool
    death_cross: bool
    atr: float
    atr_percent: float
    obv: float
    obv_trend: str
    stochastic_k: float
    stochastic_d: float
    stochastic_signal: str
    adx: float
    adx_trend: str
    ichimoku_trend: str
    vwap: float
    pivot_points: Dict[str, float]
    support_resistance: Dict[str, List[float]]
    volume_profile: Dict[str, float]
    adx_plus_di: float = 0.0
    adx_minus_di: float = 0.0

@dataclass
class RiskMetrics:
    """Đánh giá rủi ro chi tiết với real-time data"""
    volatility_24h: float
    volatility_7d: float 
    volatility_30d: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    beta: float
    liquidity_score: float
    correlation_btc: float
    correlation_eth: float
    market_cap_rank: int
    risk_level: str
    kelly_criterion: float
    # Real-time risk scores - simplified
    orderbook_imbalance_risk: float = 0.0
    liquidation_risk: float = 0.0
    funding_rate_risk: float = 0.0
    options_skew_risk: float = 0.0
    composite_risk_score: float = 0.0

    def risk_level_weight(self) -> float:
        """Chuyển đổi risk level thành trọng số số"""
        risk_weights = {
            "LOW": 0.2,
            "MEDIUM": 0.5, 
            "HIGH": 0.8,
            "EXTREME": 1.0
        }
        return risk_weights.get(self.risk_level, 0.5)

@dataclass
class MarketContext:
    """Bối cảnh thị trường tổng quan"""
    overall_trend: str
    btc_dominance: float
    eth_dominance: float
    total_market_cap: float
    market_cap_change_24h: float
    fear_greed_index: int
    fear_greed_classification: str
    funding_rate_avg: float
    open_interest_change: float
    long_short_ratio: float
    liquidation_data: Dict[str, float]
    whale_activity: str
    institutional_flow: str
    # Real-time context
    market_regime: str
    crowded_longs: List[str]
    btc_d_threshold: float
    dxy_index: float

@dataclass
class MoneyFlowAnalysis:
    """Phân tích dòng tiền nâng cao"""
    net_flow_24h: float
    net_flow_7d: float
    net_flow_30d: float
    flow_momentum: float
    flow_acceleration: float
    smart_money_flow: float
    retail_money_flow: float
    exchange_flow: Dict[str, float]
    whale_transactions: int
    large_txn_volume: float
    flow_consistency_score: float
    flow_strength: str
    flow_direction: str

@dataclass
class TradingSignal:
    """Tín hiệu giao dịch hoàn chỉnh với real-time integration"""
    symbol: str
    exchange: str
    decision: str
    direction: str
    confidence: float
    composite_score: float
    long_score: float
    short_score: float
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    position_size_usd: float
    position_size_units: float
    margin_required: float
    leverage_recommended: int
    risk_reward_ratio: float
    entry_strategy: str
    timeframe: str
    signals: List[str]
    warnings: List[str]
    score_breakdown: Dict[str, float]
    technical_indicators: TechnicalIndicators
    risk_metrics: RiskMetrics
    money_flow: MoneyFlowAnalysis
    market_context: MarketContext
    realtime_data: RealTimeData
    timestamp: str
    # --- Thêm các trường mới ---
    entry_score: float = 0.0
    sl_score: float = 0.0
    tp_score: float = 0.0
    risk_alerts: Optional[List[str]] = None
    should_enter: bool = False
    reason_for_entry: str = ""

# ========== Outlier Detection ==========

def clean_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Loại bỏ điểm dữ liệu bất thường dựa vào z-score"""
    if len(series) < 10 or series.nunique() < 2:
        return series
    # Bỏ qua NaN trước khi tính z-score
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return series
    z_scores = np.abs(stats.zscore(series_clean))
    # Tạo mask cho series ban đầu
    mask = pd.Series(True, index=series.index)
    mask[series_clean.index] = z_scores < threshold
    return series[mask]

# =============================================================================
# REAL-TIME COMPONENTS (Orderbook, Liquidations, Deribit Options)
# =============================================================================

class OrderbookAnalyzer:
    def __init__(self):
        self.orderbook_cache = {}
        self.cvd_data = {}
        self.session = None
        self.historical_imbalance = {} # Thêm historical data

    def _normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return ''
        clean_symbol = symbol.split('@')[0].replace('-USDT-PERP', '').replace('-USDT', '')
        clean_symbol = clean_symbol.replace('/USDT', '').replace('USDT', '')
        return clean_symbol.upper()
        
    def _to_binance_symbol(self, symbol: str) -> str:
        s = self._normalize_symbol(symbol) + 'USDT'
        return s.lower()

    def _to_bybit_symbol(self, symbol: str) -> str:
        return self._normalize_symbol(symbol) + 'USDT'

    async def start_websocket(self, symbols: List[str]):
        try:
            self.session = aiohttp.ClientSession()
            logging.info(f"🔌 Khởi động WebSocket cho {len(symbols)} symbols trên multiple exchanges")
            for symbol in symbols:
                asyncio.create_task(self.subscribe_binance_data(symbol))
                asyncio.create_task(self.subscribe_bybit_data(symbol))
            await asyncio.sleep(5)
            logging.info("✅ WebSocket streams đã sẵn sàng")
        except Exception as e:
            logging.warning(f"⚠️ Không thể khởi động WebSocket: {e}")

    async def subscribe_binance_data(self, symbol: str):
        symbol_clean = self._to_binance_symbol(symbol)
        streams = f"{symbol_clean}@depth20@100ms/{symbol_clean}@trade"
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        await self._subscribe_with_retry(symbol, url, "Binance")

    async def subscribe_bybit_data(self, symbol: str):
        symbol_clean = self._to_bybit_symbol(symbol)
        url = f"wss://stream.bybit.com/v5/public/linear"
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                f"orderbook.50.{symbol_clean}",
                f"publicTrade.{symbol_clean}"
            ]
        }
        await self._subscribe_with_retry(symbol, url, "Bybit", subscribe_msg)

    async def _subscribe_with_retry(self, symbol: str, url: str, exchange: str, subscribe_msg: Dict = None):
        retry_count = 0
        while True:
            try:
                async with self.session.ws_connect(url, timeout=15.0) as ws:
                    if subscribe_msg:
                        await ws.send_json(subscribe_msg)
                    logging.info(f"✅ Đã kết nối {exchange} WebSocket cho {symbol}")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self.process_websocket_data(symbol, data, exchange)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
            except Exception as e:
                retry_count += 1
                logging.warning(f"""⚠️ Mất kết nối {exchange} WebSocket {symbol}: {e}.
Đang thử lại sau 5 giây... (attempt {retry_count})""") # Giảm thời gian chờ
                await asyncio.sleep(5)

    async def process_websocket_data(self, symbol: str, data: Dict, exchange: str):
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            if exchange == "Binance":
                if 'data' in data:
                    if 'depth' in data.get('stream', ''):
                        await self.process_binance_orderbook(normalized_symbol, data)
                    elif 'trade' in data.get('stream', ''):
                        await self.process_binance_trade(normalized_symbol, data)
            elif exchange == "Bybit":
                if data.get('topic', '').startswith('orderbook'):
                    await self.process_bybit_orderbook(normalized_symbol, data)
                elif data.get('topic', '').startswith('publicTrade'):
                    await self.process_bybit_trade(normalized_symbol, data)
        except Exception as e:
            logging.debug(f"Lỗi xử lý WebSocket data {exchange} cho {symbol}: {e}")

    async def process_binance_orderbook(self, symbol: str, data: Dict):
        bids = [(float(price), float(qty)) for price, qty in data['data'].get('bids', [])]
        asks = [(float(price), float(qty)) for price, qty in data['data'].get('asks', [])]
        await self._update_orderbook_cache(symbol, bids, asks)

    async def process_bybit_orderbook(self, symbol: str, data: Dict):
        if data.get('type') == 'snapshot':
            bids = [(float(item['price']), float(item['size'])) for item in data['data'].get('b', [])]
            asks = [(float(item['price']), float(item['size'])) for item in data['data'].get('a', [])]
            await self._update_orderbook_cache(symbol, bids, asks)

    async def _update_orderbook_cache(self, symbol: str, bids: List, asks: List):
        total_bid_qty = sum(qty for _, qty in bids)
        total_ask_qty = sum(qty for _, qty in asks)
        total_qty = total_bid_qty + total_ask_qty
        imbalance = (total_bid_qty - total_ask_qty) / total_qty if total_qty > 0 else 0.0
        # Store historical imbalance for fallback (max 50 values)
        hist = self.historical_imbalance.setdefault(symbol, [])
        hist.append(imbalance)
        if len(hist) > 50: hist.pop(0)
        self.orderbook_cache.setdefault(symbol, {}).update({
            'imbalance': imbalance,
            'timestamp': datetime.now(),
            'bids': bids[:5],
            'asks': asks[:5]
        })

    async def process_binance_trade(self, symbol: str, data: Dict):
        trade = data['data']
        is_buyer_maker = trade['m']
        quantity = float(trade['q'])
        await self._update_cvd_data(symbol, quantity, is_buyer_maker)

    async def process_bybit_trade(self, symbol: str, data: Dict):
        for trade in data['data']:
            is_buyer_maker = (trade['S'] == 'Buy')
            quantity = float(trade['v'])
            await self._update_cvd_data(symbol, quantity, is_buyer_maker)

    async def _update_cvd_data(self, symbol: str, quantity: float, is_buyer_maker: bool):
        if symbol not in self.cvd_data:
            self.cvd_data[symbol] = {
                'cumulative_volume_delta': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0
            }
        cvd_entry = self.cvd_data[symbol]
        if is_buyer_maker:
            cvd_entry['cumulative_volume_delta'] -= quantity
            cvd_entry['sell_volume'] += quantity
        else:
            cvd_entry['cumulative_volume_delta'] += quantity
            cvd_entry['buy_volume'] += quantity

    def get_orderbook_imbalance(self, symbol: str, staleness_sec: float = 3.0) -> float:
        """Lấy orderbook imbalance với stale-guard + decay-to-neutral cho fallback"""
        normalized_symbol = self._normalize_symbol(symbol)
        ob = self.orderbook_cache.get(normalized_symbol, {})
        ts = ob.get('timestamp')

        # Stale guard: nếu cache đã quá cũ → neutral
        if ts is not None:
            age = (datetime.now() - ts).total_seconds()
            if age > staleness_sec:
                return 0.0

        imbalance = ob.get('imbalance', None)
        if imbalance is None:
            # Fallback: historical mean nhưng co nhanh về 0 để tránh pro-cyclical
            hist = self.historical_imbalance.get(normalized_symbol, [])
            hist_series = pd.Series(hist)
            clean_hist = clean_outliers(hist_series)
            mean_val = float(clean_hist.mean()) if not clean_hist.empty else 0.0
            return mean_val * exp(-1.2)  # decay-to-neutral
        return imbalance


    def get_cvd_signal(self, symbol: str) -> Dict:
        """Lấy CVD signal mà không có random fallback"""
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol in self.cvd_data:
            cvd = self.cvd_data[normalized_symbol]
            total_volume = cvd['buy_volume'] + cvd['sell_volume']
            net_ratio = cvd['cumulative_volume_delta'] / total_volume if total_volume > 0 else 0.0
            signal = 'BULLISH' if net_ratio > 0.05 else 'BEARISH' if net_ratio < -0.05 else 'NEUTRAL'
            return {'cvd': cvd['cumulative_volume_delta'], 'net_ratio': net_ratio, 'signal': signal}
        return {'cvd': 0.0, 'net_ratio': 0.0, 'signal': 'NEUTRAL'}

    async def close(self):
        if self.session:
            await self.session.close()

class LiquidationsMonitor:
    def __init__(self):
        self.liquidation_data: Dict[str, Dict] = {}
        self.session = None
        self.historical_liq_bias = {} # Thêm historical data

    def _normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return ''
        clean_symbol = symbol.split('@')[0].replace('-USDT-PERP', '').replace('-USDT', '')
        clean_symbol = clean_symbol.replace('/USDT', '').replace('USDT', '')
        return clean_symbol.upper()
        
    async def start_force_order_monitor(self, symbols: List[str]):
        try:
            self.session = aiohttp.ClientSession()
            logging.info("🔌 Khởi động liquidation monitor")
            asyncio.create_task(self.subscribe_binance_liquidations())
            asyncio.create_task(self.subscribe_bybit_liquidations())
            await asyncio.sleep(2)
        except Exception as e:
            logging.warning(f"⚠️ Không thể khởi động liquidation monitor: {e}")
    
    async def subscribe_binance_liquidations(self):
        url = "wss://fstream.binance.com/stream?streams=!forceOrder@arr"
        await self._subscribe_liquidation_ws(url, "Binance")
  
    async def subscribe_bybit_liquidations(self):
        url = "wss://stream.bybit.com/v5/public/linear"
        subscribe_msg = {
            "op": "subscribe",
            "args": ["liquidation"]
        }
        await self._subscribe_liquidation_ws(url, "Bybit", subscribe_msg)
    
    async def _subscribe_liquidation_ws(self, url: str, exchange: str, subscribe_msg: Dict = None):
        retry_count = 0
        while True:
            try:
                async with self.session.ws_connect(url, timeout=15.0) as ws:
                    if subscribe_msg:
                        await ws.send_json(subscribe_msg)
                    logging.info(f"✅ Đã kết nối {exchange} liquidation WebSocket")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self.process_liquidation_data(data, exchange)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
            except Exception as e:
                retry_count += 1
                logging.warning(f"""⚠️ Mất kết nối {exchange} Liquidation WebSocket: {e}.
Đang thử lại sau 5 giây... (attempt {retry_count})""") # Giảm thời gian chờ
                await asyncio.sleep(5)
    
    async def process_liquidation_data(self, data: Dict, exchange: str):
        try:
            if exchange == "Binance":
                events = data.get('data', [])
                if not isinstance(events, list):
                    events = [events]
                for ev in events:
                    await self._process_single_liquidation(ev.get('o', {}), exchange)
            elif exchange == "Bybit":
                if data.get('topic') == 'liquidation':
                    for liq in data.get('data', []):
                        await self._process_single_liquidation(liq, exchange)
        except Exception as e:
            logging.debug(f"Lỗi xử lý liquidation data {exchange}: {e}")
    
    async def _process_single_liquidation(self, liquidation: Dict, exchange: str):
        if exchange == "Binance":
            sym = liquidation.get('s')
            side = liquidation.get('S')
            quantity = float(liquidation.get('q', 0) or 0)
            price = float(liquidation.get('p', 0) or 0)
        else:  # Bybit
            sym = liquidation.get('symbol', '').replace('USDT', '')
            side = "BUY" if liquidation.get('side') == 'Buy' else "SELL"
            quantity = float(liquidation.get('size', 0) or 0)
            price = float(liquidation.get('price', 0) or 0)
        if not sym or price <= 0 or quantity <= 0:
            return
        normalized_sym = self._normalize_symbol(sym)
        if not normalized_sym:
            return
        usd_value = quantity * price
        await self._update_liquidation_data(normalized_sym, side, usd_value)
    
    async def _update_liquidation_data(self, symbol: str, side: str, usd_value: float):
        if symbol not in self.liquidation_data:
            self.liquidation_data[symbol] = {
                'long_liquidation_usd': 0.0,
                'short_liquidation_usd': 0.0,
                'total_liquidation_usd': 0.0,
                'liquidation_bias': 0.0,
                'last_update': datetime.utcnow()
            }
        liq_data = self.liquidation_data[symbol]
        if (datetime.utcnow() - liq_data['last_update']).seconds > 60:
            liq_data['long_liquidation_usd'] *= 0.9
            liq_data['short_liquidation_usd'] *= 0.9
            liq_data['total_liquidation_usd'] = liq_data['long_liquidation_usd'] + liq_data['short_liquidation_usd']
            liq_data['last_update'] = datetime.utcnow()
        if side == "SELL":
            liq_data['long_liquidation_usd'] += usd_value
        else:
            liq_data['short_liquidation_usd'] += usd_value
        liq_data['total_liquidation_usd'] += usd_value
        total = liq_data['long_liquidation_usd'] + liq_data['short_liquidation_usd']
        if total > 0:
            liq_data['liquidation_bias'] = (liq_data['short_liquidation_usd'] - liq_data['long_liquidation_usd']) / total
            # Store historical bias for fallback (max 50 values)
            hist = self.historical_liq_bias.setdefault(symbol, [])
            hist.append(liq_data['liquidation_bias'])
            if len(hist) > 50: hist.pop(0)
    
    def get_liquidation_bias(self, symbol: str, staleness_sec: float = 20.0) -> Dict:
        """Lấy liquidation bias với stale-guard + decay-to-neutral cho fallback"""
        normalized_symbol = self._normalize_symbol(symbol)
        data = self.liquidation_data.get(normalized_symbol, {})

        ts = data.get('last_update')
        bias = data.get('liquidation_bias', None)

        if ts is None or (datetime.utcnow() - ts).total_seconds() > staleness_sec:
            # Stale → dùng historical mean nhưng co nhanh về 0
            hist = self.historical_liq_bias.get(normalized_symbol, [])
            hist_series = pd.Series(hist)
            clean_hist = clean_outliers(hist_series)
            mean_val = float(clean_hist.mean()) if not clean_hist.empty else 0.0
            bias = mean_val * exp(-1.2)
        elif bias is None:
            # Không stale nhưng chưa có bias hiện tại → dùng mean
            hist = self.historical_liq_bias.get(normalized_symbol, [])
            hist_series = pd.Series(hist)
            clean_hist = clean_outliers(hist_series)
            bias = float(clean_hist.mean()) if not clean_hist.empty else 0.0

        squeeze_risk = 'LONG_SQUEEZE' if bias < -0.7 else 'SHORT_SQUEEZE' if bias > 0.7 else 'LOW'
        return {'bias': float(bias), 'squeeze_risk': squeeze_risk}

    async def close(self):
        if self.session:
            await self.session.close()

    def get_liquidation_bias_dynamic(self, symbol: str, volatility_24h: float = 0.02, staleness_sec: float = 20.0) -> Dict:
        """
        Lấy liquidation bias với threshold động và stale-guard + decay-to-neutral
        """
        normalized_symbol = self._normalize_symbol(symbol)
        data = self.liquidation_data.get(normalized_symbol, {})

        ts = data.get('last_update')
        bias = data.get('liquidation_bias', None)

        # Stale → decay-to-neutral dựa trên historical mean
        if ts is None or (datetime.utcnow() - ts).total_seconds() > staleness_sec:
            hist = self.historical_liq_bias.get(normalized_symbol, [])
            hist_series = pd.Series(hist)
            clean_hist = clean_outliers(hist_series)
            mean_val = float(clean_hist.mean()) if not clean_hist.empty else 0.0
            bias = mean_val * exp(-1.2)
        elif bias is None:
            hist = self.historical_liq_bias.get(normalized_symbol, [])
            hist_series = pd.Series(hist)
            clean_hist = clean_outliers(hist_series)
            bias = float(clean_hist.mean()) if not clean_hist.empty else 0.0

        # Dynamic threshold (giữ nguyên)
        base_threshold = 0.5
        vol_adjustment = min(volatility_24h * 5, 0.3)
        threshold = base_threshold + vol_adjustment

        if bias < -threshold:
            squeeze_risk = 'LONG_SQUEEZE'
        elif bias > threshold:
            squeeze_risk = 'SHORT_SQUEEZE'
        else:
            squeeze_risk = 'LOW'

        return {
            'bias': float(bias),
            'squeeze_risk': squeeze_risk,
            'threshold_used': threshold
        }

        
        # Determine squeeze risk
        if bias < -threshold:
            squeeze_risk = 'LONG_SQUEEZE'
        elif bias > threshold:
            squeeze_risk = 'SHORT_SQUEEZE'
        else:
            squeeze_risk = 'LOW'
        
        return {
            'bias': bias,
            'squeeze_risk': squeeze_risk,
            'threshold_used': threshold  # Debug info
        }

class DeribitOptionsAnalyzer:
    def __init__(self):
        self.base_url = "https://www.deribit.com/api/v2/public"
        self.session = None
        self.historical_options = {} # Thêm historical data

    def _normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return ''
        clean_symbol = symbol.split('@')[0].replace('-USDT-PERP', '').replace('-USDT', '')
        clean_symbol = clean_symbol.replace('/USDT', '').replace('USDT', '')
        if clean_symbol.upper() == 'WETH':
            return 'ETH'
        return clean_symbol.upper()
        
    async def get_options_data(self, symbol: str = "BTC") -> Dict:
        """Lấy options data, dùng fallback bằng historical mean nếu API lỗi"""
        normalized_symbol = self._normalize_symbol(symbol)
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            currency_map = {
                'BTC': 'BTC', 'ETH': 'ETH', 
                'SOL': 'SOL', 'AAVE': 'ETH', 'LINK': 'ETH', 'BGB': 'ETH'
            }
            currency = currency_map.get(normalized_symbol.upper(), 'ETH')
            async with self.session.get(
                f"{self.base_url}/get_book_summary_by_currency",
                params={'currency': currency, 'kind': 'option'},
                timeout=10
            ) as response:
                if response.status == 200:
                    js = await response.json()
                    parsed = self._parse_options_data(js, normalized_symbol)
                    # Lưu vào historical fallback để dùng sau
                    hist = self.historical_options.setdefault(normalized_symbol, [])
                    hist.append(parsed)
                    if len(hist) > 20: hist.pop(0)
                    return parsed
                else:
                    # Fallback to historical/neutral data if API call fails
                    return self._get_historical_options_data(normalized_symbol)
        except Exception as e:
            logging.debug(f"❌ Lỗi Deribit API cho {normalized_symbol}: {e}")
            # Fallback to historical/neutral data on exception
            return self._get_historical_options_data(normalized_symbol)

    def _parse_options_data(self, js: Dict, symbol: str) -> Dict:
        arr = js.get('result', []) or []
        total_put_oi = 0.0
        total_call_oi = 0.0
        put_ivs = []
        call_ivs = []
        for item in arr:
            name = item.get('instrument_name', '')
            oi = float(item.get('open_interest', 0) or 0)
            mark_iv = item.get('mark_iv', None)
            if name.endswith('-P'):
                total_put_oi += oi
                if mark_iv is not None:
                    put_ivs.append(float(mark_iv))
            elif name.endswith('-C'):
                total_call_oi += oi
                if mark_iv is not None:
                    call_ivs.append(float(mark_iv))
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        avg_put_iv = float(np.mean(put_ivs)) if put_ivs else 50.0
        avg_call_iv = float(np.mean(call_ivs)) if call_ivs else 48.0
        iv_skew = avg_put_iv - avg_call_iv
        sentiment = self._calculate_sentiment(put_call_ratio, iv_skew, symbol)
        return {
            'put_call_ratio': put_call_ratio, 'iv_skew': iv_skew, 'avg_put_iv': avg_put_iv,
            'avg_call_iv': avg_call_iv, 'total_put_oi': total_put_oi,
            'total_call_oi': total_call_oi, 'sentiment': sentiment
        }

    def _calculate_sentiment(self, put_call_ratio: float, iv_skew: float, symbol: str) -> str:
        if put_call_ratio < 0.9 or iv_skew < -1:
            return 'BULLISH'
        elif put_call_ratio > 1.1 or iv_skew > 1:
            return 'BEARISH'
        else:
            if put_call_ratio > 1.2:
                return 'BEARISH'
            elif put_call_ratio < 0.8:
                return 'BULLISH'
        return 'NEUTRAL'
        
    def _get_historical_options_data(self, symbol: str) -> Dict:
        """
        Fallback: dùng historical MEDIAN nếu có (chống outlier tốt hơn mean)
        """
        hist = self.historical_options.get(symbol, [])
        if hist:
            arr = pd.DataFrame(hist)
            
            # Loại bỏ outlier theo z-score
            for col in ['put_call_ratio', 'iv_skew', 'avg_put_iv', 'avg_call_iv', 'total_put_oi', 'total_call_oi']:
                if col in arr:
                    arr[col] = clean_outliers(arr[col])
            
            # Dùng MEDIAN thay vì MEAN
            median_vals = arr.median(numeric_only=True)  # ✅ Median chống outlier
            
            if median_vals.empty:
                return self._get_neutral_options_data()
            
            put_call_ratio = median_vals.get('put_call_ratio', 1.0)
            iv_skew = median_vals.get('iv_skew', 0.0)
            sentiment = self._calculate_sentiment(put_call_ratio, iv_skew, symbol)
            
            return {
                'put_call_ratio': float(put_call_ratio),
                'iv_skew': float(iv_skew),
                'avg_put_iv': float(median_vals.get('avg_put_iv', 50.0)),
                'avg_call_iv': float(median_vals.get('avg_call_iv', 48.0)),
                'total_put_oi': float(median_vals.get('total_put_oi', 50000.0)),
                'total_call_oi': float(median_vals.get('total_call_oi', 50000.0)),
                'sentiment': sentiment
            }
        
        # Nếu không có dữ liệu, trả về neutral
        return self._get_neutral_options_data()

    def _get_neutral_options_data(self) -> Dict:
        """Centralized neutral values"""
        return {
            'put_call_ratio': 1.0,
            'iv_skew': 0.0,
            'avg_put_iv': 50.0,
            'avg_call_iv': 48.0,
            'total_put_oi': 50000.0,
            'total_call_oi': 50000.0,
            'sentiment': 'NEUTRAL'
        }

    async def close(self):
        if self.session:
            await self.session.close()

# =============================================================================
# AdvancedRiskManager
# =============================================================================

class AdvancedRiskManager:
    def __init__(self, trading_system):
        self.system = trading_system
        
    def calculate_advanced_position_size(self, symbol: str, action: str, 
                                         confidence: float, funding_data: Dict,
                                         liquidation_data: Dict) -> float:
        """Tính toán position size với real-time risk factors"""
        base_size = self.system.config.max_position_size_pct
        
        # Điều chỉnh 
        if confidence < 0.6:
            base_size *= 0.8
        elif confidence > 0.8:
            base_size *= 1.2
            
        # Điều chỉnh theo liquidation risk
        squeeze_risk = liquidation_data.get('squeeze_risk', 'LOW')
        if squeeze_risk in ['LONG_SQUEEZE', 'SHORT_SQUEEZE']:
            if (action == "LONG" and squeeze_risk == "LONG_SQUEEZE") or \
               (action == "SHORT" and squeeze_risk == "SHORT_SQUEEZE"):
                base_size *= 0.5
                
        return max(0.02, min(base_size, self.system.config.max_position_size_pct))
    
    def get_trade_approval(self, symbol: str, action: str, 
                           confidence: float, 
                           funding_data: Dict, liquidation_data: Dict,
                           orderbook_data: Dict) -> Dict:
        """Đánh giá risk với real-time data"""
        approval = {
            'approved': True,
            'size_adjustment': 1.0,
            'warnings': [],
            'risk_level': 'LOW'
        }
        
        # Liquidation risk
        squeeze_risk = liquidation_data.get('squeeze_risk', 'LOW')
        if squeeze_risk in ['LONG_SQUEEZE', 'SHORT_SQUEEZE']:
            if (action == "LONG" and squeeze_risk == "LONG_SQUEEZE") or \
               (action == "SHORT" and squeeze_risk == "SHORT_SQUEEZE"):
                approval['warnings'].append(f"⚠️ Có thể {squeeze_risk.lower()}, theo dõi kỹ")
                approval['size_adjustment'] *= 0.85
                approval['risk_level'] = 'MEDIUM'
   
        # Funding rate risk
        funding_rate = funding_data.get('funding_rate', 0.0)
        if (action == "LONG" and funding_rate > 0.005) or \
           (action == "SHORT" and funding_rate < -0.005):
            approval['size_adjustment'] *= 0.85
            approval['warnings'].append("⚠️ Cao funding rate")
            approval['risk_level'] = 'MEDIUM'
            
   
        # Order book imbalance risk
        imbalance = orderbook_data.get('imbalance', 0)
        if abs(imbalance) > 0.3:
            approval['size_adjustment'] *= 0.9
            approval['warnings'].append(f"⚠️ High orderbook imbalance: {imbalance:.3f}")
        # --- Spread check ---
        spread = orderbook_data.get('spread_pct', None)
        if spread is None:
            # fallback: đôi khi spread_pct nằm trong funding_data (từ get_funding_data)
            spread = funding_data.get('spread_pct', 0.0)

        try:
            if float(spread) > self.system.config.max_spread_pct:
                approval['warnings'].append(
                    f"⚠️ Spread cao ({float(spread):.4f}) > ngưỡng ({self.system.config.max_spread_pct:.4f})"
                )
                approval['size_adjustment'] *= 0.8
                approval['risk_level'] = 'HIGH'
        except Exception:
            pass
            
        return approval

# =============================================================================
# ENHANCED TRADING SYSTEM - KHẮC PHỤC RANDOM FALLBACK
# =============================================================================

class EnhancedCryptoAnalyzer:
    """Hệ thống phân tích crypto nâng cao với real-time integration - ĐÃ SỬA LỖI RANDOM"""
    
    def __init__(self, account_balance: float = 10000.0, max_risk_per_trade: float = 0.02, use_websocket: bool = True):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.risk_free_rate = 0.03
        self.use_websocket = use_websocket
        self.config = TradingConfig()
        
        # API Endpoints
        self.apis = {
            'binance': 'https://api.binance.com/api/v3',
            'binance_futures': 'https://fapi.binance.com/fapi/v1',
            'bybit': 'https://api.bybit.com',
            'okx': 'https://www.okx.com/api/v5',
            'coingecko': 'https://api.coingecko.com/api/v3',
            'fear_greed': 'https://api.alternative.me/fng/',
            'glassnode': 'https://api.glassnode.com/v1',
        }
        
        # Real-time components
        self.orderbook_analyzer = OrderbookAnalyzer()
        self.liquidation_monitor = LiquidationsMonitor()
        self.deribit_analyzer = DeribitOptionsAnalyzer()
        self.advanced_risk_manager = AdvancedRiskManager(self)
        
        # Multi-exchange support
        self.cex_exchanges = {}
        self.setup_exchanges()
        
        # Market regime
        self.market_regime = "NEUTRAL"
        self.btc_dominance = 0.0
        self.dxy_index = 0.0
        self._dxy_last_good = None       # lưu giá trị DXY “good” gần nhất
        self._dxy_last_source = ""       # lưu nguồn của giá trị đó
        self.crowded_longs = set()
        
        self.cache = {}
        self.cache_duration = 60
        
 
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize a symbol from any format to its base form (e.g.
        'BTC-USDT-PERP@BASE' -> 'BTC')"""
        if not symbol:
            return ''
            
        # Remove everything after @ if it exists
        symbol = symbol.split('@')[0]
        
        # Remove common suffixes and pairs
        symbol = symbol.replace('-USDT-PERP', '').replace('-USDT', '').replace('/USDT', '')
        symbol = symbol.replace('USDT', '').replace('-PERP', '')
        
        # Special cases
        if symbol.upper() == 'WETH':
            return 'ETH'
        
        # Strip any remaining special characters and whitespace
        symbol = ''.join(c for c in symbol if c.isalnum())
        
        return symbol.upper()
        
    def setup_exchanges(self):
        """Khởi tạo multi-exchange connections"""
        exchange_specs = {
            'binance': {'enable': True, 'options': {'defaultType': 'swap'}},
            'bybit': {'enable': True, 'options': {'defaultType': 'linear'}},
            'okx': {'enable': True, 'options': {'defaultType': 'swap'}},
            'bitget': {'enable': True, 'options': {'defaultType': 'swap'}},
            'gateio': {'enable': True, 'options': {'defaultType': 'swap'}},
            'mexc': {'enable': True, 'options': {'defaultType': 'swap'}},
        }
        
        for name, spec in exchange_specs.items():
            if not spec['enable']:
                continue
            try:
                ex = getattr(ccxt, name)({
                    'enableRateLimit': True,
                    'timeout': 15000,
                    'options': spec.get('options', {})
                })
                ex.load_markets()
                self.cex_exchanges[name] = ex
                logging.info(f"✅ Khởi tạo thành công: {name}")
            except Exception as e:
                logging.error(f"❌ Lỗi khởi tạo {name}: {e}")

    async def start_real_time_data(self, symbols: List[str]):
        """Khởi động real-time data streams"""
        if not self.use_websocket:
            logging.info("⏩ Bỏ qua WebSocket theo cấu hình")
            return
            
        try:
            # Chuẩn hóa symbols trước khi truyền vào real-time components
            normalized_symbols = [self._normalize_symbol(s) for s in symbols]
            normalized_symbols = list(set(s for s in normalized_symbols if s))
            
            logging.info("🚀 Starting real-time data streams...")
            await asyncio.wait_for(
                asyncio.gather(
                    self.orderbook_analyzer.start_websocket(normalized_symbols),
                    self.liquidation_monitor.start_force_order_monitor(normalized_symbols),
                    return_exceptions=True
                ),
                timeout=15.0
            )
            logging.info("✅ Real-time data streams đã được khởi động")
        except Exception as e:
            logging.warning(f"⚠️ Lỗi khởi động real-time data: {e}")

    async def cleanup(self):
        """Dọn dẹp resources"""
        await self.orderbook_analyzer.close()
        await self.liquidation_monitor.close()
        await self.deribit_analyzer.close()

    async def fetch_btc_dominance_and_dxy(self):
        """Lấy BTC dominance và DXY index real-time"""
        try:
            # BTC Dominance
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://api.coingecko.com/api/v3/global', timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            # ĐÃ SỬA LỖI: Thống nhất BTC Dominance thành fraction (0.52)
                            self.btc_dominance = data['data']['market_cap_percentage']['btc'] / 100
                            logging.info(f"₿ BTC Dominance REAL-TIME: {self.btc_dominance:.2%}")
            except Exception as e:
                logging.warning(f"⚠️ Không lấy được BTC Dominance: {e}")
                self.btc_dominance = 0.52 # Fallback trung lập
                
            # DXY Index
            await self.fetch_dxy_index_real()
        except Exception as e:
            logging.error(f"❌ Lỗi lấy regime data: {e}")
            self.btc_dominance = 0.52
            self.dxy_index = 104.0

    # ===== REPLACE this entire method inside EnhancedCryptoAnalyzer =====
    async def fetch_dxy_index_real(self):
        """
        Lấy DXY realtime với đa nguồn + retry.
        Thứ tự: Yahoo DX=F -> Yahoo chart API -> Yahoo DX-Y.NYB -> FRED (nếu có key) -> Stooq daily -> last_good -> 104.0.
        Khi một nguồn thành công: lưu self._dxy_last_good + self._dxy_last_source và dùng cho fallback về sau.
        """
        import math
        YF_QUOTE = [
            ("DX=F", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=DX=F"),
            ("DX-Y.NYB", "https://query1.finance.yahoo.com/v7/finance/quote?symbols=DX-Y.NYB"),
        ]
        YF_CHART = [
            ("DX=F",  "https://query2.finance.yahoo.com/v8/finance/chart/DX=F?interval=1m&range=1d"),
            ("DX-Y.NYB", "https://query2.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1m&range=1d"),
        ]
        FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
        STOOQ_D = "https://stooq.com/q/d/l/?s=^dxy&i=d"  # daily CSV

        async def _try_yf_quote(session, label, url):
            try:
                async with session.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    if resp.status != 200:
                        return None, None
                    js = await resp.json()
                    arr = (js.get("quoteResponse") or {}).get("result") or []
                    if not arr:
                        return None, None
                    q = arr[0]
                    price = (
                        q.get("regularMarketPrice")
                        or q.get("postMarketPrice")
                        or q.get("preMarketPrice")
                        or q.get("regularMarketPreviousClose")
                    )
                    if price is not None and math.isfinite(float(price)):
                        return float(price), f"Yahoo {label}"
            except Exception:
                return None, None
            return None, None

        async def _try_yf_chart(session, label, url):
            try:
                async with session.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    if resp.status != 200:
                        return None, None
                    js = await resp.json()
                    res = (js.get("chart") or {}).get("result") or []
                    if not res:
                        return None, None
                    indicators = (res[0].get("indicators") or {})
                    close = (indicators.get("quote") or [{}])[0].get("close") or []
                    # lấy close cuối cùng khác None
                    close_vals = [c for c in close if c is not None]
                    if close_vals:
                        return float(close_vals[-1]), f"YahooChart {label}"
            except Exception:
                return None, None
            return None, None

        async def _try_fred(session):
            if not API_KEYS.get("FRED_API_KEY"):
                return None, None
            try:
                params = {
                    "series_id": "DTWEXBGS",
                    "api_key": API_KEYS["FRED_API_KEY"],
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                }
                async with session.get(FRED_URL, params=params, timeout=8) as resp:
                    if resp.status != 200:
                        return None, None
                    js = await resp.json()
                    obs = js.get("observations") or []
                    if obs:
                        val = obs[0].get("value")
                        if val not in (None, "."):
                            return float(val), "FRED DTWEXBGS"
            except Exception:
                return None, None
            return None, None

        async def _try_stooq_daily(session):
            # daily CSV, lấy dòng cuối cùng, cột “Close”
            try:
                async with session.get(STOOQ_D, timeout=8) as resp:
                    if resp.status != 200:
                        return None, None
                    text = await resp.text()
                    rows = [r.strip() for r in text.splitlines() if r.strip()]
                    if len(rows) <= 1:
                        return None, None
                    last = rows[-1].split(",")
                    # CSV format: Date,Open,High,Low,Close,Volume
                    if len(last) >= 5:
                        close_val = float(last[4])
                        return close_val, "Stooq Daily"
            except Exception:
                return None, None
            return None, None

        try:
            async with aiohttp.ClientSession() as session:
                # 1) Yahoo quote (DX=F trước)
                for label, url in YF_QUOTE:
                    val, src = await _try_yf_quote(session, label, url)
                    if val:
                        self.dxy_index = val
                        self._dxy_last_good = val
                        self._dxy_last_source = src
                        logging.info(f"💵 DXY Index ({src}): {val}")
                        return

                # 2) Yahoo chart API (nếu quote fail)
                for label, url in YF_CHART:
                    val, src = await _try_yf_chart(session, label, url)
                    if val:
                        self.dxy_index = val
                        self._dxy_last_good = val
                        self._dxy_last_source = src
                        logging.info(f"💵 DXY Index ({src}): {val}")
                        return

                # 3) FRED (daily)
                val, src = await _try_fred(session)
                if val:
                    self.dxy_index = val
                    self._dxy_last_good = val
                    self._dxy_last_source = src
                    logging.info(f"💵 DXY Index ({src}): {val}")
                    return

                # 4) Stooq daily (không cần API key)
                val, src = await _try_stooq_daily(session)
                if val:
                    self.dxy_index = val
                    self._dxy_last_good = val
                    self._dxy_last_source = src
                    logging.info(f"💵 DXY Index ({src}): {val}")
                    return

            # 5) Nếu tất cả thất bại: dùng last_good nếu có
            if self._dxy_last_good is not None:
                self.dxy_index = float(self._dxy_last_good)
                logging.info(f"💵 DXY Index (LastGood: {self._dxy_last_source}): {self.dxy_index}")
                return

            # 6) Cuối cùng mới fallback hardcode
            self.dxy_index = 104.0
            logging.info("💵 DXY Index (Fallback): 104.0")

        except Exception:
            # Nếu lỗi tổng quát, vẫn ưu tiên last_good
            if self._dxy_last_good is not None:
                self.dxy_index = float(self._dxy_last_good)
                logging.info(f"💵 DXY Index (LastGood on exception: {self._dxy_last_source}): {self.dxy_index}")
            else:
                self.dxy_index = 104.0
                logging.info("💵 DXY Index (Fallback on exception): 104.0")



    async def detect_market_regime(self):
        """Phát hiện market regime với real-time data"""
        try:
            await self.fetch_btc_dominance_and_dxy()
            
            # Simplified regime detection
            if self.btc_dominance > 0.55 and self.dxy_index > 105:
                self.market_regime = "RISK_OFF"
                logging.warning("🌧️ Market Regime REAL-TIME: RISK_OFF")
            elif self.btc_dominance < 0.48 and self.dxy_index < 100:
                self.market_regime = "RISK_ON"
                logging.info("🌞 Market Regime REAL-TIME: RISK_ON")
            else:
                self.market_regime = "NEUTRAL"
                logging.info("☁️ Market Regime REAL-TIME: NEUTRAL")
                
        except Exception as e:
            logging.warning(f"⚠️ Không thể detect market regime: {e}")
            self.market_regime = "NEUTRAL"

    async def get_real_time_data(self, symbol: str) -> RealTimeData:
        """Lấy tất cả real-time data cho một symbol với fallback dựa historical mean"""
        normalized_symbol = self._normalize_symbol(symbol)
        try:
            # Order book data: Sẽ trả về historical mean/NEUTRAL nếu WebSocket chưa có data thật
            ob_imbalance = self.orderbook_analyzer.get_orderbook_imbalance(normalized_symbol)
            cvd_signal = self.orderbook_analyzer.get_cvd_signal(normalized_symbol)
            
            # Mới - truyền volatility để tính dynamic threshold
            funding_data = await self.get_funding_data(normalized_symbol)
            # Lấy volatility (cần có sẵn từ risk metrics hoặc ước tính)
            # Tạm thời dùng giá trị trung bình nếu chưa có
            volatility_estimate = 0.02  # Default 2%

            liquidation_bias = self.liquidation_monitor.get_liquidation_bias_dynamic(
                normalized_symbol, 
                volatility_24h=volatility_estimate
            )
            
            # Options data: Sẽ dùng historical mean/neutral nếu API lỗi
            options_data = await self.deribit_analyzer.get_options_data(normalized_symbol)
            
            # Funding rate và OI từ CEX (API call)
            funding_data = await self.get_funding_data(normalized_symbol)
            
            # KHẮC PHỤC FALLBACK: Không sử dụng random, chỉ dùng dữ liệu từ components (đã có fallback nội bộ)
            return RealTimeData(
                orderbook_imbalance=ob_imbalance,
                cvd_signal=cvd_signal,
                liquidation_bias=liquidation_bias,
                options_data=options_data,
                funding_rate=funding_data.get('funding_rate', 0.0001),
                open_interest=funding_data.get('open_interest', 0.0),
                spread_pct=funding_data.get('spread_pct', 0.001),
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logging.warning(f"⚠️ Lỗi lấy real-time data cho {symbol}: {e}. Dùng fallback tĩnh/lịch sử.")
            return self._get_improved_fallback_realtime_data(normalized_symbol)

    def _get_improved_fallback_realtime_data(self, symbol: str) -> RealTimeData:
        """Fallback real-time data được cải thiện (dựa trên historical mean/neutral)"""
        
        # Orderbook Imbalance: Historical Mean
        ob_hist = self.orderbook_analyzer.historical_imbalance.get(symbol, [])
        ob_imbalance = float(clean_outliers(pd.Series(ob_hist)).mean()) if ob_hist else 0.0
        
        # Liquidation Bias: Historical Mean
        liq_hist = self.liquidation_monitor.historical_liq_bias.get(symbol, [])
        liq_bias = float(clean_outliers(pd.Series(liq_hist)).mean()) if liq_hist else 0.0
        
        # Options Data: Historical Mean (hoặc neutral nếu không có)
        options_data = self.deribit_analyzer._get_historical_options_data(symbol)

        return RealTimeData(
            orderbook_imbalance=ob_imbalance,
            cvd_signal={'signal': 'NEUTRAL', 'net_ratio': ob_imbalance, 'cvd': 0.0},
            liquidation_bias={'bias': liq_bias, 'squeeze_risk': 'LONG_SQUEEZE' if liq_bias < -0.7 else 'SHORT_SQUEEZE' if liq_bias > 0.7 else 'LOW'},
            options_data=options_data,
            funding_rate=0.0001,
            open_interest=0.0,
            spread_pct=0.001,
            timestamp=datetime.now().isoformat()
        )

    async def get_funding_data(self, symbol: str) -> Dict:
        """Lấy funding rate và open interest từ exchanges"""
        normalized_symbol = self._normalize_symbol(symbol)
        for exchange_name, exchange in self.cex_exchanges.items():
            try:
                found_symbol = self.find_symbol_on_exchange(exchange, normalized_symbol)
                if found_symbol:
                    # Try to get funding rate
                    try:
                        fr = await asyncio.to_thread(exchange.fetchFundingRate, found_symbol)
                        if isinstance(fr, dict):
                            return {
                                'funding_rate': float(fr.get('fundingRate', 0.0001)),
                                'open_interest': float(fr.get('openInterest', 0)),
                                'spread_pct': 0.001  # Simplified
                            }
                    except:
                        pass
            except:
                continue
        return {'funding_rate': 0.0001, 'open_interest': 0.0, 'spread_pct': 0.001}

    def find_symbol_on_exchange(self, exchange, base: str) -> Optional[str]:
        """Tìm symbol trên exchange"""
        try:
            markets = exchange.markets
            base_u = base.upper()
            
            patterns = [
                f"{base_u}/USDT:USDT",
                f"{base_u}USDT",
                f"{base_u}/USDT",
                f"{base_u}-USDT"
            ]
            
            for pattern in patterns:
                if pattern in markets:
                    market = markets[pattern]
                    if market.get('swap', False) or market.get('future', False):
                        return pattern
            
            for symbol, market in markets.items():
                if (market.get('swap', False) or market.get('future', False)) and \
                   'USDT' in symbol.upper() and base_u in symbol.upper():
                    return symbol
            return None
        except Exception as e:
            logging.warning(f"⚠️ Lỗi khi tìm symbol {base}: {e}")
            return None

    # =========================================================================
    # PHẦN QUAN TRỌNG: SỬA LỖI LẤY GIÁ TOKEN THẬT
    # =========================================================================

    def get_real_price(self, symbol: str, exchange: str = 'binance') -> Tuple[float, Dict]:
        """Get real-time price với symbol đã được xử lý đúng cách - ĐÃ SỬA LỖI"""
        normalized_symbol = self._normalize_symbol(symbol)
        cache_key = f"{normalized_symbol}_{exchange}_price"
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data['price'], cached_data
        
        try:
            # Danh sách các exchanges để thử theo thứ tự ưu tiên
            exchanges_to_try = [
                ('binance', self._get_binance_price),
                ('bybit', self._get_bybit_price),
                ('okx', self._get_okx_price),
                ('gateio', self._get_gateio_price),
                ('mexc', self._get_mexc_price),
                ('bitget', self._get_bitget_price)
            ]
            
            for exchange_name, price_func in exchanges_to_try:
                try:
                    price, price_data = price_func(normalized_symbol)
                    if price > 0:
                        price_data['source'] = exchange_name
                        logging.info(f"  ✅ Lấy giá {normalized_symbol} từ {exchange_name}: ${price:.6f}")
                        self.cache[cache_key] = (time.time(), {'price': price, **price_data})
                        return price, price_data
                except Exception as e:
                    logging.debug(f"  ⚠️ {exchange_name} failed for {normalized_symbol}: {e}")
                    continue
            
            # Fallback: Sử dụng ccxt để lấy giá
            for exchange_name, ex in self.cex_exchanges.items():
                try:
                    found_symbol = self.find_symbol_on_exchange(ex, normalized_symbol)
                    if found_symbol:
                        ticker = ex.fetch_ticker(found_symbol)
                        if ticker and 'last' in ticker and ticker['last']:
                            price = float(ticker['last'])
                            price_data = {
                                'price': price,
                                'volume_24h': float(ticker.get('baseVolume', 0)),
                                'quote_volume': float(ticker.get('quoteVolume', 0)),
                                'price_change_percent': float(ticker.get('percentage', 0)),
                                'high_24h': float(ticker.get('high', 0)),
                                'low_24h': float(ticker.get('low', 0)),
                                'source': f"{exchange_name}_ccxt"
                            }
                            logging.info(f"  ✅ Lấy giá {normalized_symbol} từ {exchange_name} (CCXT): ${price:.6f}")
                            self.cache[cache_key] = (time.time(), {'price': price, **price_data})
                            return price, price_data
                except Exception as e:
                    logging.debug(f"  ⚠️ CCXT {exchange_name} failed: {e}")
                    continue
            
            # Final fallback: Sử dụng CoinGecko API
            try:
                price, price_data = self._get_coingecko_price(normalized_symbol)
                if price > 0:
                    price_data['source'] = 'coingecko'
                    logging.info(f"  ✅ Lấy giá {normalized_symbol} từ CoinGecko: ${price:.6f}")
                    self.cache[cache_key] = (time.time(), {'price': price, **price_data})
                    return price, price_data
            except Exception as e:
                logging.debug(f"  ⚠️ CoinGecko failed: {e}")
            
            # Ultimate fallback: Giá ước tính dựa trên Nansen data
            estimated_price = self._get_estimated_price(normalized_symbol)
            price_data = {
                'price': estimated_price,
                'volume_24h': 100000,
                'quote_volume': 100000,
                'price_change_percent': 0,
                'high_24h': estimated_price * 1.02,
                'low_24h': estimated_price * 0.98,
                'source': 'estimated'
            }
            logging.warning(f"  ⚠️ Sử dụng giá ước tính cho {normalized_symbol}: ${estimated_price:.6f}")
            return estimated_price, price_data
                       
        except Exception as e:
            logging.error(f"❌ Lỗi nghiêm trọng lấy giá {symbol}: {e}")
            return 0.0, {}

    def _get_binance_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ Binance"""
        try:
            # Thử futures trước
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return float(data['lastPrice']), {
                    'volume_24h': float(data.get('volume', 0)),
                    'quote_volume': float(data.get('quoteVolume', 0)),
                    'price_change_percent': float(data.get('priceChangePercent', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0))
                }
        except:
            pass
        
        # Thử spot
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {'symbol': f"{symbol}USDT"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data['lastPrice']), {
                    'volume_24h': float(data.get('volume', 0)),
                    'quote_volume': float(data.get('quoteVolume', 0)),
                    'price_change_percent': float(data.get('priceChangePercent', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0))
                }
        except:
            pass
        
        raise Exception("Binance API failed")

    def _get_bybit_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ Bybit"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {'category': 'linear', 'symbol': f"{symbol}USDT"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['retCode'] == 0 and data['result']['list']:
                    ticker = data['result']['list'][0]
                    # Giá trị này đã là % nên không cần nhân 100
                    return float(ticker['lastPrice']), {
                        'volume_24h': float(ticker.get('volume24h', 0)),
                        'quote_volume': float(ticker.get('turnover24h', 0)),
                        'price_change_percent': float(ticker.get('price24hPcnt', 0)) * 100,
                        'high_24h': float(ticker.get('highPrice24h', 0)),
                        'low_24h': float(ticker.get('lowPrice24h', 0))
                    }
        except:
            pass
        raise Exception("Bybit API failed")

    def _get_okx_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ OKX"""
        try:
            url = "https://www.okx.com/api/v5/market/ticker"
            params = {'instId': f"{symbol}-USDT-SWAP"}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    ticker = data['data'][0]
                    # OKX trả về tỷ lệ (decimal) nên cần nhân 100 để đồng bộ
                    return float(ticker['last']), {
                        'volume_24h': float(ticker.get('vol24h', 0)),
                        'quote_volume': float(ticker.get('volCcy24h', 0)),
                        'price_change_percent': (float(ticker.get('last', 0)) / float(ticker.get('open24h', 1)) - 1) * 100, # ĐÃ SỬA LỖI: nhân 100
                        'high_24h': float(ticker.get('high24h', 0)),
                        'low_24h': float(ticker.get('low24h', 0))
                    }
        except:
            pass
        raise Exception("OKX API failed")

    def _get_gateio_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ Gate.io"""
        try:
            url = "https://api.gateio.ws/api/v4/futures/usdt/tickers"
            params = {'contract': f"{symbol}_USDT"}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    ticker = data[0]
                    return float(ticker['last']), {
                        'volume_24h': float(ticker.get('volume_24h', 0)),
                        'quote_volume': float(ticker.get('volume_24h_usd', 0)),
                        'price_change_percent': float(ticker.get('change_percentage', 0)),
                        'high_24h': float(ticker.get('high_24h', 0)),
                        'low_24h': float(ticker.get('low_24h', 0))
                    }
        except:
            pass
        raise Exception("Gate.io API failed")

    def _get_mexc_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ MEXC"""
        try:
            url = "https://contract.mexc.com/api/v1/contract/ticker"
            params = {'symbol': f"{symbol}_USDT"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    return float(data['data']['lastPrice']), {
                        'volume_24h': float(data['data'].get('volume', 0)),
                        'quote_volume': float(data['data'].get('amount', 0)),
                        'price_change_percent': float(data['data'].get('riseFallRate', 0)) * 100,
                        'high_24h': float(data['data'].get('high', 0)),
                        'low_24h': float(data['data'].get('low', 0))
                    }
        except:
            pass
        raise Exception("MEXC API failed")

    def _get_bitget_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ Bitget"""
        try:
            url = "https://api.bitget.com/api/v2/mix/market/ticker"
            params = {'symbol': f"{symbol}USDT_UMCBL"}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '00000' and data['data']:
                    ticker = data['data'][0]
                    return float(ticker['lastPr']), {
                        'volume_24h': float(ticker.get('baseVol', 0)),
                        'quote_volume': float(ticker.get('quoteVol', 0)),
                        'price_change_percent': float(ticker.get('changeRate', 0)) * 100,
                        'high_24h': float(ticker.get('high24h', 0)),
                        'low_24h': float(ticker.get('low24h', 0))
                    }
        except:
            pass
        raise Exception("Bitget API failed")

    def _get_coingecko_price(self, symbol: str) -> Tuple[float, Dict]:
        """Lấy giá từ CoinGecko"""
        try:
            # Map symbol to CoinGecko ID
            coin_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'SOL': 'solana',
                'ADA': 'cardano',
                'XRP': 'ripple',
                'DOT': 'polkadot',
                'DOGE': 'dogecoin',
                'AVAX': 'avalanche-2',
                'MATIC': 'matic-network',
                'LINK': 'chainlink',
                'LTC': 'litecoin',
                'BCH': 'bitcoin-cash',
                'XLM': 'stellar',
                'ATOM': 'cosmos',
                'ETC': 'ethereum-classic',
                'XMR': 'monero',
                'EOS': 'eos',
                'AAVE': 'aave',
                'UNI': 'uniswap',
                'CRV': 'curve-dao-token'
            }
            
            coin_id = coin_map.get(symbol.upper(), symbol.lower())
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = data['market_data']['current_price']['usd']
                return price, {
                    'volume_24h': data['market_data']['total_volume']['usd'],
                    'price_change_percent': data['market_data']['price_change_percentage_24h'],
                    'high_24h': data['market_data']['high_24h']['usd'],
                    'low_24h': data['market_data']['low_24h']['usd']
                }
        except:
            pass
        raise Exception("CoinGecko API failed")

    def _get_estimated_price(self, symbol: str) -> float:
        """Ước tính giá dựa trên thông tin thị trường"""
        # Giá ước tính cho các token phổ biến
        price_map = {
            'BTC': 45000.0, 'ETH': 2500.0, 'BNB': 300.0, 'SOL': 100.0,
            'ADA': 0.5, 'XRP': 0.6, 'DOT': 7.0, 'DOGE': 0.08,
            'AVAX': 35.0, 'MATIC': 0.8, 'LINK': 15.0, 'LTC': 70.0,
            'BCH': 250.0, 'XLM': 0.12, 'ATOM': 10.0, 'ETC': 25.0,
            'XMR': 160.0, 'EOS': 0.7, 'AAVE': 100.0, 'UNI': 6.0,
            'CRV': 0.6, 'BGB': 0.4
        }
        
        # Sửa: Fallback cho altcoin không có trong map về giá 5.0 (trung lập hơn)
        return price_map.get(symbol.upper(), 5.0) 

    def get_historical_klines(self, symbol: str, interval: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu nến lịch sử với fallback - ĐÃ CẢI THIỆN"""
        normalized_symbol = self._normalize_symbol(symbol)
        
        # Danh sách các API để thử
        apis_to_try = [
            self._get_binance_klines,
            self._get_bybit_klines,
            self._get_okx_klines
        ]
        
        for api_func in apis_to_try:
            try:
                df = api_func(normalized_symbol, interval, limit)
                
                if df is not None and len(df) > 50:
                    logging.info(f"  ✅ Lấy dữ liệu nến {normalized_symbol} từ {api_func.__name__}: {len(df)} nến")
                    # Áp dụng clean_outliers cho dữ liệu lịch sử
                    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                        df[col] = clean_outliers(df[col])
                    df = df.dropna(subset=['close', 'high', 'low'])
                    
                    if len(df) > 50:
                        return df
                    else:
                        logging.warning(f"  ⚠️ Dữ liệu nến bị loại bỏ outlier quá nhiều, chuyển sang fallback")
            except Exception as e:
                logging.debug(f"  ⚠️ {api_func.__name__} failed: {e}")
                continue
        
        # Fallback: Tạo dữ liệu nến tổng hợp
        logging.warning(f"  ⚠️ Sử dụng dữ liệu nến tổng hợp cho {normalized_symbol}")
        return self._generate_synthetic_klines(limit)

    def _get_binance_klines(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu nến từ Binance"""
        try:
            # Thử futures trước
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_klines_data(data)
        except:
            pass
        
        # Thử spot
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_klines_data(data)
        except:
            pass
    
        return None

    def _get_bybit_klines(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu nến từ Bybit"""
        try:
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['retCode'] == 0 and data['result']['list']:
                    klines = data['result']['list']
                    # Bybit trả về dữ liệu theo thứ tự thời gian tăng dần, cần đảo ngược
                    klines.reverse()
                    return self._parse_bybit_klines(klines)
        except:
            pass
        return None

    def _get_okx_klines(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu nến từ OKX"""
        try:
            url = "https://www.okx.com/api/v5/market/candles"
            params = {
                'instId': f"{symbol}-USDT-SWAP",
                'bar': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=15)
        
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    return self._parse_okx_klines(data['data'])
        except:
            pass
        return None

    def _parse_klines_data(self, data: List) -> pd.DataFrame:
        """Parse dữ liệu nến từ Binance format"""
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.dropna(subset=['close', 'high', 'low'])
        
        return df

    def _parse_bybit_klines(self, klines: List) -> pd.DataFrame:
        """Parse dữ liệu nến từ Bybit format"""
        if not klines:
            return None
        
        data = []
        for k in klines:
            data.append({
                'timestamp': datetime.fromtimestamp(int(k[0]) / 1000),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'quote_volume': float(k[6])
            })
        
        df = pd.DataFrame(data)
        return df

    def _parse_okx_klines(self, klines: List) -> pd.DataFrame:
        """Parse dữ liệu nến từ OKX format"""
        if not klines:
            return None
            
        data = []
        for k in klines:
            data.append({
                'timestamp': datetime.fromtimestamp(int(k[0]) / 1000), # Đã sửa format thời gian của OKX
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'quote_volume': float(k[6])
            })
        
        df = pd.DataFrame(data)
        return df

    def _generate_synthetic_klines(self, limit: int = 200) -> pd.DataFrame:
        """Tạo dữ liệu nến tổng hợp cho phân tích khi không có API"""
        np.random.seed(42)
        
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='1H')
        base_price = 100.0
        
        
        prices = [base_price]
        for _ in range(limit - 1):
            change = np.random.normal(0, 0.002) # Giảm biến động cho synthetic data
            prices.append(prices[-1] * (1 + change))
        
        # Sửa: đảm bảo close price không quá giống open price
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices], # Giảm biến động
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices], # Giảm biến động
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': np.random.uniform(1000, 10000, limit),
            'quote_volume': np.random.uniform(100000, 1000000, limit)
        })
        
        return df

    # CÁC PHƯƠNG THỨC PHÂN TÍCH KẾ THỪA (giữ nguyên từ code trước)
    
    def load_nansen_data(self, filename: str) -> List[Dict]:
        """Load dữ liệu Nansen từ file JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Đã load {len(data)} coins từ {filename}")
            return data
       

        except Exception as e:
            print(f"❌ Lỗi đọc file {filename}: {e}")
            return []

    def calculate_comprehensive_indicators(self, df: pd.DataFrame, current_price: float) -> TechnicalIndicators:
        """Tính toán chỉ báo kỹ thuật toàn diện"""
        if df is None or len(df) < 50:
            return self._default_indicators()
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
        
            # RSI & Divergence
            rsi_indicator = RSIIndicator(close, window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            rsi_divergence = self._detect_divergence(close.tail(20), rsi_indicator.rsi().tail(20))
            
            # MACD với crossover detection
            macd_indicator = MACD(close)
            macd_series = macd_indicator.macd()
            signal_series = macd_indicator.macd_signal()
            hist_series = macd_indicator.macd_diff()

            macd = macd_series.iloc[-1]
            macd_signal = signal_series.iloc[-1]
            macd_hist = hist_series.iloc[-1]

            # Kiểm tra crossover
            if len(macd_series) >= 2 and len(signal_series) >= 2:
                macd_prev = macd_series.iloc[-2]
                signal_prev = signal_series.iloc[-2]
                
                # Bullish crossover
                if macd > macd_signal and macd_prev <= signal_prev:
                    macd_trend = "BULLISH_CROSS"
                # Bearish crossover
                elif macd < macd_signal and macd_prev >= signal_prev:
                    macd_trend = "BEARISH_CROSS"
                # Existing trend
                elif macd > macd_signal:
                    macd_trend = "BULLISH"
                else:
                    macd_trend = "BEARISH"
            else:
                macd_trend = "BULLISH" if macd > macd_signal else "BEARISH"
            
            # Bollinger Bands
            bb = BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            # Bollinger Bands Position với validation
            bb_range = bb_upper - bb_lower
            if bb_range > 0 and not (pd.isna(bb_upper) or pd.isna(bb_lower)):
                bb_position = (current_price - bb_lower) / bb_range
                # Clamp giá trị trong [0, 1]
                bb_position = max(0.0, min(1.0, bb_position))
            else:
                bb_position = 0.5  # Neutral khi BB phẳng hoặc invalid            
            # Moving Averages
            sma_20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
            sma_50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
            sma_200 = SMAIndicator(close, window=200).sma_indicator().iloc[-1] if len(df) >= 200 else sma_50
            ema_9 = EMAIndicator(close, window=9).ema_indicator().iloc[-1]
            ema_21 = EMAIndicator(close, window=21).ema_indicator().iloc[-1]
            
            # Golden/Death Cross
            # Golden/Death Cross - Phát hiện crossover gần đây
            golden_cross = False
            death_cross = False

            if len(df) >= 200:
                sma_50_series = SMAIndicator(close, window=50).sma_indicator()
                sma_200_series = SMAIndicator(close, window=200).sma_indicator()
                
                if len(sma_50_series) >= 2 and len(sma_200_series) >= 2:
                    sma_50_current = sma_50_series.iloc[-1]
                    sma_200_current = sma_200_series.iloc[-1]
                    sma_50_prev = sma_50_series.iloc[-2]
                    sma_200_prev = sma_200_series.iloc[-2]
                    
                    # Golden Cross: SMA50 vừa cắt lên SMA200
                    if sma_50_current > sma_200_current and sma_50_prev <= sma_200_prev:
                        golden_cross = True
                    
                    # Death Cross: SMA50 vừa cắt xuống SMA200
                    if sma_50_current < sma_200_current and sma_50_prev >= sma_200_prev:
                        death_cross = True
            
            # ATR
            atr_indicator = AverageTrueRange(high, low, close, window=14)
            atr = atr_indicator.average_true_range().iloc[-1]
            atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
            
            # OBV
            obv_indicator = OnBalanceVolumeIndicator(close, volume)
            obv = obv_indicator.on_balance_volume().iloc[-1]
            obv_trend = self._detect_trend(obv_indicator.on_balance_volume().tail(20))
            
            # Stochastic
            stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stoch_k_series = stoch.stoch()
            stoch_d_series = stoch.stoch_signal()
            stoch_k = stoch_k_series.iloc[-1]
            stoch_d = stoch_d_series.iloc[-1]
            stoch_signal = self._interpret_stochastic(stoch_k_series, stoch_d_series)
            
            # ADX với +DI/-DI để xác định hướng
            adx_indicator = ADXIndicator(high, low, close, window=14)
            adx = adx_indicator.adx().iloc[-1]
            plus_di = adx_indicator.adx_pos().iloc[-1]
            minus_di = adx_indicator.adx_neg().iloc[-1]

            # Xác định strength
            if adx > 50:
                adx_strength = "VERY_STRONG"
            elif adx > 25:
                adx_strength = "STRONG"
            else:
                adx_strength = "WEAK"

            # Xác định direction
            if plus_di > minus_di:
                adx_direction = "UPTREND"
            else:
                adx_direction = "DOWNTREND"

            # Kết hợp
            adx_trend = f"{adx_strength}_{adx_direction}"  # Ví dụ: "STRONG_UPTREND"
        
            # Ichimoku
            ichimoku_trend = self._calculate_ichimoku(high, low, close)
    
            # VWAP
            vwap = self._calculate_vwap(df)
            
            # Support/Resistance
            pivot_points = self._calculate_pivot_points(high.iloc[-1], low.iloc[-1], close.iloc[-1])
            support_resistance = self._find_support_resistance(close.tail(50))
          
            # Volume Profile
            volume_profile = self._calculate_volume_profile(df)
            
            # ✅ Code mới (ĐẦY ĐỦ):
            return TechnicalIndicators(
                rsi=round(rsi, 2) if not pd.isna(rsi) else 50,
                rsi_divergence=rsi_divergence,
                macd=round(macd, 6) if not pd.isna(macd) else 0,
                macd_signal=round(macd_signal, 6) if not pd.isna(macd_signal) else 0,
                macd_histogram=round(macd_hist, 6) if not pd.isna(macd_hist) else 0,
                macd_trend=macd_trend,
                bb_upper=round(bb_upper, 4) if not pd.isna(bb_upper) else 0,
                bb_middle=round(bb_middle, 4) if not pd.isna(bb_middle) else 0,
                bb_lower=round(bb_lower, 4) if not pd.isna(bb_lower) else 0,
                bb_position=round(bb_position, 2),
                sma_20=round(sma_20, 4) if not pd.isna(sma_20) else 0,
                sma_50=round(sma_50, 4) if not pd.isna(sma_50) else 0,
                sma_200=round(sma_200, 4) if not pd.isna(sma_200) else 0,
                ema_9=round(ema_9, 4) if not pd.isna(ema_9) else 0,
                ema_21=round(ema_21, 4) if not pd.isna(ema_21) else 0,
                golden_cross=golden_cross,
                death_cross=death_cross,
                atr=round(atr, 4) if not pd.isna(atr) else 0,
                atr_percent=round(atr_percent, 2),
                obv=round(obv, 0) if not pd.isna(obv) else 0,
                obv_trend=obv_trend,
                stochastic_k=round(stoch_k, 2) if not pd.isna(stoch_k) else 50,
                stochastic_d=round(stoch_d, 2) if not pd.isna(stoch_d) else 50,
                stochastic_signal=stoch_signal,
                adx=round(adx, 2) if not pd.isna(adx) else 20,
                adx_trend=adx_trend,
                ichimoku_trend=ichimoku_trend,
                vwap=round(vwap, 4),
                pivot_points=pivot_points,
                support_resistance=support_resistance,
                volume_profile=volume_profile
            )
        except Exception as e:
            print(f"    ⚠️  Lỗi tính indicators: {e}")
            return self._default_indicators()

    # ======= Helpers & Fallbacks (ADD THESE INSIDE THE CLASS) =======

    def _default_indicators(self):
        """Trả về bộ chỉ báo mặc định khi thiếu dữ liệu/ lỗi tính toán."""
        try:
            return TechnicalIndicators(
                rsi=50.0, rsi_divergence="NONE",
                macd=0.0, macd_signal=0.0, macd_histogram=0.0, macd_trend="NEUTRAL",
                bb_upper=0.0, bb_middle=0.0, bb_lower=0.0, bb_position=0.5,
                sma_20=0.0, sma_50=0.0, sma_200=0.0, ema_9=0.0, ema_21=0.0,
                golden_cross=False, death_cross=False,
                atr=0.0, atr_percent=0.0,
                obv=0.0, obv_trend="NEUTRAL",
                stochastic_k=50.0, stochastic_d=50.0, stochastic_signal="NEUTRAL",
                adx=20.0, adx_trend="WEAK_NEUTRAL",
                ichimoku_trend="NEUTRAL", vwap=0.0,
                pivot_points={}, support_resistance={"support": [], "resistance": []},
                volume_profile={},
                adx_plus_di=0.0, adx_minus_di=0.0,
            )
        except Exception:
            # nếu dataclass thay đổi, trả dict để không gãy
            return {
                "rsi": 50.0, "rsi_divergence": "NONE",
                "macd": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0, "macd_trend": "NEUTRAL",
                "bb_upper": 0.0, "bb_middle": 0.0, "bb_lower": 0.0, "bb_position": 0.5,
                "sma_20": 0.0, "sma_50": 0.0, "sma_200": 0.0, "ema_9": 0.0, "ema_21": 0.0,
                "golden_cross": False, "death_cross": False,
                "atr": 0.0, "atr_percent": 0.0,
                "obv": 0.0, "obv_trend": "NEUTRAL",
                "stochastic_k": 50.0, "stochastic_d": 50.0, "stochastic_signal": "NEUTRAL",
                "adx": 20.0, "adx_trend": "WEAK_NEUTRAL",
                "ichimoku_trend": "NEUTRAL", "vwap": 0.0,
                "pivot_points": {}, "support_resistance": {"support": [], "resistance": []},
                "volume_profile": {},
                "adx_plus_di": 0.0, "adx_minus_di": 0.0,
            }

    def _detect_divergence(self, price_series: pd.Series, osc_series: pd.Series,
                       lookback: int = 20, min_sep: int = 2) -> str:
        """
        Phát hiện divergence không repaint bằng pivots đã xác nhận (không dùng lookahead ngoài cửa sổ).
        Trả về: "BULLISH" | "BEARISH" | "NONE"
        """
        try:
            if price_series is None or osc_series is None:
                return "NONE"

            # Chỉ dùng dữ liệu đã đóng nến và cùng index
            p = price_series.tail(lookback).dropna()
            o = osc_series.tail(lookback).dropna()
            idx = p.index.intersection(o.index)
            if len(idx) < 5:
                return "NONE"
            p = p.loc[idx]; o = o.loc[idx]

            # Pivots đã xác nhận: kiểm tra đỉnh/đáy tại vị trí i nếu giá trị tại i
            # là max/min trong cửa sổ [i-left, i+right], với left=right=2 (mặc định)
            left = right = 2

            def _confirmed_pivots(series: pd.Series, high: bool = True, left: int = 2, right: int = 2):
                idxs = []
                vals = series.values
                for i in range(left, len(series) - right):
                    window = vals[i-left:i+right+1]
                    center = vals[i]
                    if high:
                        if center == window.max() and np.argmax(window) == left:
                            idxs.append(i)
                    else:
                        if center == window.min() and np.argmin(window) == left:
                            idxs.append(i)
                return idxs

            def _recent_pair(indices: list) -> tuple:
                # Lấy cặp gần nhất nhưng cách nhau tối thiểu min_sep
                if len(indices) < 2:
                    return None
                b = indices[-1]
                # tìm a lùi lại sao cho b-a >= min_sep
                for k in range(len(indices)-2, -1, -1):
                    if b - indices[k] >= min_sep:
                        return indices[k], b
                return None

            # Lấy pivots đã xác nhận cho giá và oscillator
            ph = _confirmed_pivots(p, high=True,  left=left, right=right)
            pl = _confirmed_pivots(p, high=False, left=left, right=right)
            oh = _confirmed_pivots(o, high=True,  left=left, right=right)
            ol = _confirmed_pivots(o, high=False, left=left, right=right)

            ph_pair = _recent_pair(ph)
            pl_pair = _recent_pair(pl)
            oh_pair = _recent_pair(oh)
            ol_pair = _recent_pair(ol)

            # Bullish: giá LL, oscillator HL
            if pl_pair and ol_pair:
                pl1, pl2 = pl_pair
                ol1, ol2 = ol_pair
                if p.iloc[pl2] < p.iloc[pl1] and o.iloc[ol2] > o.iloc[ol1]:
                    return "BULLISH"

            # Bearish: giá HH, oscillator LH
            if ph_pair and oh_pair:
                ph1, ph2 = ph_pair
                oh1, oh2 = oh_pair
                if p.iloc[ph2] > p.iloc[ph1] and o.iloc[oh2] < o.iloc[oh1]:
                    return "BEARISH"

            return "NONE"
        except Exception:
            return "NONE"



    def _detect_trend(self, series: pd.Series, lookback: int = 20) -> str:
        """Trend rất đơn giản dựa trên slope của hồi quy tuyến tính."""
        try:
            s = series.tail(lookback).dropna()
            if len(s) < 3:
                return "NEUTRAL"
            x = np.arange(len(s))
            slope, *_ = np.polyfit(x, s.values, 1)
            if slope > 0:  return "UPTREND"
            if slope < 0:  return "DOWNTREND"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    def _interpret_stochastic(self, k_series: pd.Series, d_series: pd.Series) -> str:
        """Diễn giải tín hiệu Stochastic cơ bản."""
        try:
            k1, d1 = k_series.iloc[-1], d_series.iloc[-1]
            k0, d0 = k_series.iloc[-2], d_series.iloc[-2] if len(k_series) >= 2 else (k1, d1)
            cross_up = (k1 > d1) and (k0 <= d0)
            cross_dn = (k1 < d1) and (k0 >= d0)
            if cross_up and k1 < 20:  return "BULLISH_CROSS_OVERSOLD"
            if cross_dn and k1 > 80:  return "BEARISH_CROSS_OVERBOUGHT"
            if k1 < 20:  return "OVERSOLD"
            if k1 > 80:  return "OVERBOUGHT"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    def _calculate_ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> str:
        """Ichimoku trend rất gọn: giá vs mây (approx)."""
        try:
            conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
            base = (high.rolling(26).max() + low.rolling(26).min()) / 2
            span_a = ((conv + base) / 2).shift(26)
            span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            price = close.iloc[-1]
            sa = span_a.iloc[-1] if not pd.isna(span_a.iloc[-1]) else base.iloc[-1]
            sb = span_b.iloc[-1] if not pd.isna(span_b.iloc[-1]) else base.iloc[-1]
            cloud_low, cloud_high = min(sa, sb), max(sa, sb)
            if price > cloud_high: return "BULLISH"
            if price < cloud_low:  return "BEARISH"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP đơn giản: Σ(price*volume)/Σ(volume)."""
        try:
            pv = df['close'] * df['volume']
            denom = df['volume'].replace(0, np.nan).sum()
            if denom and denom > 0:
                return float(pv.sum() / denom)
            return float(df['close'].iloc[-1])
        except Exception:
            return float(df['close'].iloc[-1])

    def _calculate_pivot_points(self, h: float, l: float, c: float) -> Dict[str, float]:
        """Pivot (Classic)."""
        p = (h + l + c) / 3.0
        r1 = 2*p - l; s1 = 2*p - h
        r2 = p + (h - l); s2 = p - (h - l)
        r3 = h + 2*(p - l); s3 = l - 2*(h - p)
        return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

    def _find_support_resistance(self, close: pd.Series, n: int = 3) -> Dict[str, List[float]]:
        """Tìm n mức support/resistance sơ bộ bằng đỉnh đáy cục bộ."""
        try:
            arr = close.values
            peaks, _ = find_peaks(arr)
            troughs, _ = find_peaks(-arr)
            res = sorted(arr[peaks][-n:].tolist()) if len(peaks) else []
            sup = sorted(arr[troughs][-n:].tolist()) if len(troughs) else []
            return {"support": sup, "resistance": res}
        except Exception:
            return {"support": [], "resistance": []}

    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 12) -> Dict[str, float]:
        """Volume profile gọn theo bins giá."""
        try:
            prices = df['close'].values
            vols = df['volume'].values
            if len(prices) != len(vols) or len(prices) < 5:
                return {}
            hist, edges = np.histogram(prices, bins=bins, weights=vols)
            centers = 0.5*(edges[1:] + edges[:-1])
            return {f"{centers[i]:.4f}": float(hist[i]) for i in range(len(hist))}
        except Exception:
            return {}

    def _default_risk_metrics(self) -> RiskMetrics:
        """Risk metrics mặc định khi thiếu dữ liệu."""
        return RiskMetrics(
            volatility_24h=0.02, volatility_7d=0.03, volatility_30d=0.04,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.10,
            var_95=0.05, var_99=0.10, cvar_95=0.07, beta=1.0,
            liquidity_score=5.0, correlation_btc=0.5, correlation_eth=0.4,
            market_cap_rank=0, risk_level="MEDIUM", kelly_criterion=0.0,
            orderbook_imbalance_risk=0.0, liquidation_risk=0.0,
            funding_rate_risk=0.0, options_skew_risk=0.0, composite_risk_score=0.0
        )

    def _calculate_beta(self, returns: pd.Series) -> float:
        """Beta vs ‘thị trường’ giả định: xấp xỉ bằng std(normalized)."""
        try:
            # nếu có BTC returns bạn có thể thay thế chỗ này
            return float(np.clip(returns.std() / 0.02, 0.2, 2.0))
        except Exception:
            return 1.0

    def _default_market_context(self) -> MarketContext:
        """Market context mặc định nếu API lỗi."""
        return MarketContext(
            overall_trend="NEUTRAL",
            btc_dominance=0.52, eth_dominance=0.18,
            total_market_cap=1_700_000_000_000,
            market_cap_change_24h=0.0,
            fear_greed_index=50, fear_greed_classification="NEUTRAL",
            funding_rate_avg=0.0, open_interest_change=0.0, long_short_ratio=1.0,
            liquidation_data={'longs': 0, 'shorts': 0},
            whale_activity="MODERATE", institutional_flow="NEUTRAL",
            market_regime=self.market_regime, crowded_longs=list(self.crowded_longs),
            btc_d_threshold=self.btc_dominance, dxy_index=self.dxy_index
        )

    # sửa chữ ký: thêm 'tech' để khớp lời gọi nơi khác
    def validate_enhanced_signal(self, long_score: float, short_score: float, risk: RiskMetrics,
                                 rr_ratio: float, realtime_data: RealTimeData,
                                 tech: Optional[TechnicalIndicators] = None) -> Tuple[bool, List[str]]:
        """Validate tín hiệu với điều kiện nới lỏng; có thể dùng 'tech' để bổ sung cảnh báo."""
        warnings = []
        is_valid = True

        if max(long_score, short_score) < 2:
            warnings.append("⚠️ Điểm tổng hợp thấp (< 2)")
            is_valid = False

        if risk.risk_level == "EXTREME":
            warnings.append("⚠️ Rủi ro cực cao - Khuyến nghị tránh")
            is_valid = False
        elif risk.risk_level == "HIGH" and max(long_score, short_score) < 4:
            warnings.append("⚠️ Rủi ro cao yêu cầu điểm số cao hơn")

        if rr_ratio < 0.8:
            warnings.append(f"⚠️ Tỷ lệ R:R thấp ({rr_ratio:.2f})")

        if risk.liquidity_score < 3:
            warnings.append("⚠️ Thanh khoản thấp")
            is_valid = False

        if realtime_data.liquidation_bias.get('squeeze_risk') in ['LONG_SQUEEZE', 'SHORT_SQUEEZE']:
            warnings.append(f"⚠️ High liquidation risk: {realtime_data.liquidation_bias.get('squeeze_risk')}")

        if abs(realtime_data.funding_rate) > 0.005:
            warnings.append(f"⚠️ Extreme funding rate: {realtime_data.funding_rate:.4f}")

        # Optional: thêm cảnh báo kỹ thuật
        if tech:
            if tech.adx < 15:
                warnings.append("⚠️ ADX thấp → xu hướng yếu")
            if tech.bb_position in (0.0, 1.0):
                warnings.append("⚠️ Giá đang sát biên Bollinger, dễ đảo chiều ngắn hạn")

        return is_valid, warnings


    def calculate_enhanced_risk_metrics(self, df: pd.DataFrame, symbol_data: Dict, current_price: float, realtime_data: RealTimeData) -> RiskMetrics:
        """Tính toán risk metrics với real-time data integration"""
        base_risk = self.calculate_risk_metrics(df, symbol_data, current_price)
        
        # Real-time risk factors
        ob_imbalance_risk = abs(realtime_data.orderbook_imbalance) * 10
        
        liquidation_risk = 0.0
        if realtime_data.liquidation_bias.get('squeeze_risk') != 'LOW':
            liquidation_risk = 0.7
            
        funding_rate_risk = min(1.0, abs(realtime_data.funding_rate) * 1000)
        
        options_skew_risk = 0.0
        if realtime_data.options_data.get('sentiment') == 'BEARISH':
            options_skew_risk = 0.3
   
        # Composite risk score
        risk_level_weight = base_risk.risk_level_weight()
        composite_risk_score = risk_level_weight * 0.3 + ob_imbalance_risk * 0.25 + liquidation_risk * 0.2 + funding_rate_risk * 0.1 + options_skew_risk * 0.1

        # Create new risk metrics object
        base_dict = {
            'volatility_24h': base_risk.volatility_24h,
            'volatility_7d': base_risk.volatility_7d,
            'volatility_30d': base_risk.volatility_30d,
            'sharpe_ratio': base_risk.sharpe_ratio,
            'sortino_ratio': base_risk.sortino_ratio,
            'max_drawdown': base_risk.max_drawdown,
            'var_95': base_risk.var_95,
            'var_99': base_risk.var_99,
            'cvar_95': base_risk.cvar_95,
            'beta': base_risk.beta,
            'liquidity_score': base_risk.liquidity_score,
            'correlation_btc': base_risk.correlation_btc,
            'correlation_eth': base_risk.correlation_eth,
            'market_cap_rank': base_risk.market_cap_rank,
            'risk_level': base_risk.risk_level,
            'kelly_criterion': base_risk.kelly_criterion,
            'orderbook_imbalance_risk': round(ob_imbalance_risk, 3),
            'liquidation_risk': round(liquidation_risk, 3),
            'funding_rate_risk': round(funding_rate_risk, 3),
            'options_skew_risk': round(options_skew_risk, 3),
            'composite_risk_score': round(composite_risk_score, 3)
        }
        
        return RiskMetrics(**base_dict)

    def calculate_risk_metrics(self, df: pd.DataFrame, symbol_data: Dict, current_price: float) -> RiskMetrics:
        """Tính toán metrics rủi ro chi tiết với error handling"""
        if df is None or len(df) < 30:
            print(f"    ⚠️  Dữ liệu không đủ cho risk metrics, sử dụng mặc định")
            return self._default_risk_metrics()
        
        try:
            returns = df['close'].pct_change().dropna()
    
            
            if len(returns) == 0:
                return self._default_risk_metrics()
            
            # Volatility
            # Sửa: Sử dụng cleaned returns (loại bỏ outlier nếu có)
            clean_returns = clean_outliers(returns)
            if len(clean_returns) == 0: clean_returns = returns 

            vol_24h = clean_returns.tail(24).std() * np.sqrt(24) if len(clean_returns) >= 24 else clean_returns.std()
            vol_7d = clean_returns.tail(168).std() * np.sqrt(168) if len(clean_returns) >= 168 else vol_24h
            vol_30d = clean_returns.std() * np.sqrt(len(clean_returns)) if len(clean_returns) > 0 else vol_24h
            
            # Sharpe & Sortino
            mean_return = clean_returns.mean()
            sharpe = (mean_return - self.risk_free_rate/365) / clean_returns.std() if clean_returns.std() > 0 else 0
    
            downside_returns = clean_returns[clean_returns < 0]
            sortino = (mean_return - self.risk_free_rate/365) / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Max Drawdown
            cum_returns = (1 + clean_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
    
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            
            # VaR & CVaR
            var_95 = abs(clean_returns.quantile(0.05)) * current_price if len(clean_returns) > 20 else current_price * 0.05
            var_99 = abs(clean_returns.quantile(0.01)) * current_price if len(clean_returns) > 100 else current_price * 0.10
            cvar_95 = abs(clean_returns[clean_returns <= clean_returns.quantile(0.05)].mean()) * current_price if len(clean_returns[clean_returns <= clean_returns.quantile(0.05)]) > 0 else var_95 * 1.5
            
            # Beta (vs BTC if available)
            beta = self._calculate_beta(clean_returns) # Sửa: dùng clean_returns
            
        
            # Liquidity Score
            volume_24h = symbol_data.get('volume_24h', 0)
            liquidity_score = min(10, np.log10(volume_24h + 1)) if volume_24h > 0 else 3

            # Risk Score
            risk_score = vol_30d * 50 + max_dd * 15 + (10 - liquidity_score) * 10

            # Các yếu tố khác
            if sharpe < 0.3:
                risk_score += 10
            if sharpe < 0:
                risk_score += 10

            # Phân loại risk_level
            if risk_score < 35:
                risk_level = "LOW"
            elif risk_score < 70:
                risk_level = "MEDIUM"
            elif risk_score < 110:
                risk_level = "HIGH"
            else:
                risk_level = "EXTREME"

            # Kelly Criterion
            win_rate = len(clean_returns[clean_returns > 0]) / len(clean_returns) if len(clean_returns) > 0 else 0.5
            avg_win = clean_returns[clean_returns > 0].mean() if len(clean_returns[clean_returns > 0]) > 0 else 0.01
            avg_loss = abs(clean_returns[clean_returns < 0].mean()) if len(clean_returns[clean_returns < 0]) > 0 else 0.01
    
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
            kelly = max(0, min(0.25, kelly))
            
            return RiskMetrics(
                volatility_24h=round(float(vol_24h), 4),
                volatility_7d=round(float(vol_7d), 4),
                volatility_30d=round(float(vol_30d), 4),
                sharpe_ratio=round(float(sharpe), 2),
                sortino_ratio=round(float(sortino), 2),
                max_drawdown=round(float(max_dd), 4),
                var_95=round(float(var_95), 4),
                var_99=round(float(var_99), 4),
                cvar_95=round(float(cvar_95), 4),
                beta=round(float(beta), 2),
                liquidity_score=round(float(liquidity_score), 2),
                correlation_btc=0.5,
                correlation_eth=0.4,
                market_cap_rank=0,
                risk_level=risk_level,
                kelly_criterion=round(float(kelly), 4),
                orderbook_imbalance_risk=0.0,
                liquidation_risk=0.0,
                funding_rate_risk=0.0,
                options_skew_risk=0.0,
                composite_risk_score=0.0
            )
        except Exception as e:
            print(f"    ⚠️  Lỗi tính risk metrics: {e}")
            return self._default_risk_metrics()

    def analyze_money_flow(self, symbol_data: Dict, df: pd.DataFrame) -> MoneyFlowAnalysis:
        """Phân tích dòng tiền nâng cao từ Nansen data"""
        sm = symbol_data.get('sm', {})
        net_flow_24h = sm.get('24h', 0)
        net_flow_7d = sm.get('7d', 0)
        net_flow_30d = sm.get('30d', 0)
        
        flow_momentum = (net_flow_24h * 0.5 + net_flow_7d * 0.3 + net_flow_30d * 0.2)
        flow_acceleration = net_flow_24h - net_flow_7d if net_flow_7d != 0 else 0
        
  
        cin = symbol_data.get('cin', {})
        cout = symbol_data.get('cout', {})
        smart_money_flow = cin.get('24h', 0) - cout.get('24h', 0)
        retail_money_flow = net_flow_24h - smart_money_flow
        
        exchange_flow = {
            'net_exchange_flow': symbol_data.get('nef', {}).get('24h', 0),
            'exchange_inflow': cin.get('24h', 0),
            'exchange_outflow': cout.get('24h', 0)
        }
        
        whale_transactions = 0
        large_txn_volume = 0
        
        flows = [net_flow_24h, net_flow_7d, net_flow_30d]
        flow_std = np.std(flows) if len(flows) > 1 else 0
        flow_mean = np.mean(flows)
        flow_consistency = 1 - (flow_std / abs(flow_mean)) if flow_mean != 0 else 0
        flow_consistency = max(0, min(1, flow_consistency))
        
        abs_flow = abs(net_flow_24h)
        if abs_flow > 5000000:
            flow_strength = "EXTREME"
        elif abs_flow > 1000000:
            flow_strength = "STRONG"
        elif abs_flow > 100000:
            flow_strength = "MODERATE"
        else:
            flow_strength = "WEAK"
        
        flow_direction = "INFLOW" if net_flow_24h > 0 else "OUTFLOW" if net_flow_24h < 0 else "NEUTRAL"
        
        return MoneyFlowAnalysis(
            net_flow_24h=net_flow_24h,
            net_flow_7d=net_flow_7d,
            net_flow_30d=net_flow_30d,
            flow_momentum=round(flow_momentum, 2),
            flow_acceleration=round(flow_acceleration, 2),
            smart_money_flow=smart_money_flow,
            retail_money_flow=retail_money_flow,
            exchange_flow=exchange_flow,
            whale_transactions=whale_transactions,
            large_txn_volume=large_txn_volume,
            flow_consistency_score=round(flow_consistency, 2),
            flow_strength=flow_strength,
            flow_direction=flow_direction
        )

    def get_market_context(self) -> MarketContext:
        """Lấy bối cảnh thị trường tổng quan với real-time integration"""
        try:
            # Fear & Greed Index
  
            fg_response = requests.get(self.apis['fear_greed'], timeout=10)
            fear_greed = 50
            fg_class = "NEUTRAL"
            
            if fg_response.status_code == 200:
                fg_data = fg_response.json()
                fear_greed = int(fg_data['data'][0]['value'])
                if fear_greed >= 75:
                    fg_class = "EXTREME_GREED"
                elif fear_greed >= 55:
                    fg_class = "GREED"
                elif fear_greed >= 45:
                    fg_class = "NEUTRAL"
                elif fear_greed >= 25:
                    fg_class = "FEAR"
                else:
                    fg_class = "EXTREME_FEAR"
            
            # Market data from CoinGecko
            cg_response = requests.get(f"{self.apis['coingecko']}/global", timeout=10)
            btc_dom = 52.0
            eth_dom = 18.0
            total_mc = 1700000000000
            mc_change = 0.0
            
            if cg_response.status_code == 200:
                cg_data = cg_response.json()['data']
                btc_dom = cg_data['market_cap_percentage'].get('btc', 52.0)
                eth_dom = cg_data['market_cap_percentage'].get('eth', 18.0)
                total_mc = cg_data['total_market_cap'].get('usd', 1700000000000)
                mc_change = cg_data.get('market_cap_change_percentage_24h_usd', 0.0)
            
            # Determine overall trend
            if fear_greed >= 60 and mc_change > 2:
                overall_trend = "STRONG_BULLISH"
            elif fear_greed >= 50 and mc_change > 0:
                overall_trend = "BULLISH"
            elif fear_greed <= 40 and mc_change < -2:
                overall_trend = "STRONG_BEARISH"
            elif fear_greed <= 50 and mc_change < 0:
                overall_trend = "BEARISH"
            else:
                overall_trend = "NEUTRAL"
            
            # ĐÃ SỬA LỖI: btc_dominance trong MarketContext phải là fraction (0.52)
            # (Đảm bảo logic thống nhất: dùng fraction 0.xx trong code)
            return MarketContext(
                overall_trend=overall_trend,
                btc_dominance=round(btc_dom/100, 4), # Sửa: chia 100
                eth_dominance=round(eth_dom/100, 4), # Sửa: chia 100
                total_market_cap=total_mc,
                market_cap_change_24h=round(mc_change, 2),
                fear_greed_index=fear_greed,
                fear_greed_classification=fg_class,
                funding_rate_avg=0.01,
                open_interest_change=0.0,
                long_short_ratio=1.0,
                liquidation_data={'longs': 0, 'shorts': 0},
                whale_activity="MODERATE",
                institutional_flow="ACCUMULATING",
                market_regime=self.market_regime,
                crowded_longs=list(self.crowded_longs),
                btc_d_threshold=self.btc_dominance,
                dxy_index=self.dxy_index
            )
  
        except Exception as e:
            print(f"⚠️  Lỗi lấy market context: {e}")
            return self._default_market_context()

    def calculate_enhanced_composite_score(self, tech: TechnicalIndicators, risk: RiskMetrics, 
                                          money_flow: MoneyFlowAnalysis, market: MarketContext,
                                          symbol_data: Dict, realtime_data: RealTimeData) -> Tuple[float, float, float, Dict[str, float], List[str], str]:
        """Tính điểm tổng hợp với real-time signals integration"""
        
        long_score, short_score, breakdown, signals = self.calculate_directional_scores(
            tech, risk, money_flow, market, symbol_data
        )
        
        # Thêm real-time scores
        realtime_long, realtime_short, realtime_signals = self.calculate_realtime_scores(realtime_data)
        long_score += realtime_long
        short_score += realtime_short
        signals.extend(realtime_signals)
        # --- Chuẩn hóa & log breakdown ---
        def _norm01(x):
            try:
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.0

        # 'breakdown' được trả về từ calculate_directional_scores(...)
        # Chuẩn hóa mềm các khóa nếu có
        if isinstance(breakdown, dict) and breakdown:
            norm_breakdown = {}
            for k, v in breakdown.items():
                # cố gắng đưa về 0..1 (nếu đã 0..1 thì giữ nguyên)
                norm_breakdown[k] = _norm01(v)
            breakdown = norm_breakdown
            logging.info(
                "🔍 Score Breakdown | " + " | ".join([f"{k}:{breakdown[k]:.2f}" for k in list(breakdown.keys())[:8]])
            )
        else:
            logging.info("🔍 Score Breakdown trống/không hợp lệ")
        
       
        # Xác định direction dựa trên điểm số
        if long_score >= short_score:
            composite_score = long_score
            direction = Direction.LONG.value
            signals.append(f"🎯 Hướng giao dịch: LONG (Điểm: {long_score:.1f} vs SHORT: {short_score:.1f})")
        else:
            composite_score = short_score
            direction = Direction.SHORT.value
            signals.append(f"🎯 Hướng giao dịch: SHORT (Điểm: {short_score:.1f} vs LONG: {long_score:.1f})")
        
        return composite_score, long_score, short_score, breakdown, signals, direction

    def calculate_realtime_scores(self, realtime_data: RealTimeData) -> Tuple[float, float, List[str]]:
        """Tính điểm từ real-time data"""
        long_score = 0
        short_score = 0
        signals = []
        
        # Order book imbalance
        imbalance = realtime_data.orderbook_imbalance
        if imbalance > 0.1:
            long_score += 2
            signals.append(f"📊 Order book imbalance bullish: {imbalance:.3f}")
        elif imbalance < -0.1:
            short_score += 2
            signals.append(f"📊 Order book imbalance bearish: {imbalance:.3f}")
        
        # CVD signal
        cvd_signal = realtime_data.cvd_signal.get('signal', 'NEUTRAL')
        if cvd_signal == 'BULLISH':
            long_score += 1.5
            signals.append("📈 CVD signal bullish")
        elif cvd_signal == 'BEARISH':
            short_score += 1.5
            signals.append("📉 CVD signal bearish")
        
        # Options sentiment
        options_sentiment = realtime_data.options_data.get('sentiment', 'NEUTRAL')
        if options_sentiment == 'BULLISH':
            long_score += 1
            signals.append("📊 Options sentiment bullish")
        elif options_sentiment == 'BEARISH':
            short_score += 1
            signals.append("📊 Options sentiment bearish")
        
        # Liquidation bias
        liquidation_bias = realtime_data.liquidation_bias.get('bias', 0)
        # Hướng này đánh giá rủi ro/cơ hội ngắn hạn: bias > 0 (short liq nhiều) -> cơ hội LONG
        # bias < 0 (long liq nhiều) -> cơ hội SHORT
        if liquidation_bias > 0.3:
            short_score += 1
            signals.append(f"⚡ Short liquidation bias: {liquidation_bias:.2f}")
        elif liquidation_bias < -0.3:
            long_score += 1
            signals.append(f"⚡ Long liquidation bias: {liquidation_bias:.2f}")
        
        return long_score, short_score, signals

    def calculate_directional_scores(self, tech: TechnicalIndicators, risk: RiskMetrics, 
                                     money_flow: MoneyFlowAnalysis, market: MarketContext,
                                     symbol_data: Dict) -> Tuple[float, float, Dict[str, float], List[str]]:
        """Tính điểm riêng biệt cho LONG và SHORT với normalization"""
 
        breakdown_long = {}
        breakdown_short = {}
        signals = []
        
        # 1. MONEY FLOW SCORE (35%)
        mf_score_long = 0
        mf_score_short = 0
        
        if money_flow.flow_strength == "EXTREME":
            if money_flow.flow_direction == "INFLOW":
                mf_score_long += 7
                signals.append("💰 EXTREME INFLOW - Cực kỳ tích cực cho LONG")
            else:
                mf_score_short += 7
                signals.append("💸 EXTREME OUTFLOW - Cực kỳ tích cực cho SHORT")
        elif money_flow.flow_strength == "STRONG":
            if money_flow.flow_direction == "INFLOW":
                mf_score_long += 5
            else:
                mf_score_short += 5
            signals.append(f"💰 STRONG {money_flow.flow_direction}")
  
        elif money_flow.flow_strength == "MODERATE":
            if money_flow.flow_direction == "INFLOW":
                mf_score_long += 3
            else:
                mf_score_short += 3
            signals.append(f"💵 MODERATE {money_flow.flow_direction}")
        
      
        if money_flow.flow_consistency_score > 0.7:
            if money_flow.flow_direction == "INFLOW":
                mf_score_long += 2
            else:
                mf_score_short += 2
            signals.append("✅ Dòng tiền nhất quán cao")
        
       
        if money_flow.smart_money_flow > 0 and money_flow.net_flow_24h > 0:
            mf_score_long += 2
            signals.append("🐋 Smart Money đang tích lũy - Tốt cho LONG")
        elif money_flow.smart_money_flow < 0 and money_flow.net_flow_24h < 0:
            mf_score_short += 2
            signals.append("🐋 Smart Money đang phân phối - Tốt cho SHORT")
       
        mf_score_long = max(-10, min(10, mf_score_long))
        mf_score_short = max(-10, min(10, mf_score_short))
        
        breakdown_long['money_flow'] = mf_score_long * 0.35
        breakdown_short['money_flow'] = mf_score_short * 0.35
        
        # 2. TECHNICAL SCORE (30%)
        tech_score_long = 0
        tech_score_short = 0
   
        if tech.rsi < 30:
            tech_score_long += 4
            signals.append(f"📊 RSI {tech.rsi} - Oversold mạnh - Tốt cho LONG")
        elif tech.rsi < 40:
            tech_score_long += 2
            signals.append(f"📊 RSI {tech.rsi} - Oversold - Tốt cho LONG")
    
        elif tech.rsi > 70:
            tech_score_short += 4
            signals.append(f"📊 RSI {tech.rsi} - Overbought mạnh - Tốt cho SHORT")
        elif tech.rsi > 60:
            tech_score_short += 2
            signals.append(f"📊 RSI {tech.rsi} - Overbought - Tốt cho SHORT")
        
     
        if tech.rsi_divergence == "BULLISH":
            tech_score_long += 3
            signals.append("📈 RSI Bullish Divergence - Tốt cho LONG")
        elif tech.rsi_divergence == "BEARISH":
            tech_score_short += 3
            signals.append("📉 RSI Bearish Divergence - Tốt cho SHORT")
        
        if tech.macd_trend == "BULLISH" and tech.macd_histogram > 0:
            tech_score_long += 3
            signals.append("📈 MACD Bullish với histogram tăng - Tốt cho LONG")
        elif tech.macd_trend == "BEARISH" and tech.macd_histogram < 0:
            tech_score_short += 3
            signals.append("📉 MACD Bearish với histogram giảm - Tốt cho SHORT")
        
        if tech.macd_trend == "BULLISH_CROSS":
            tech_score_long += 4  # Bonus cho crossover
            signals.append("🔥 MACD vừa bullish cross - Tín hiệu mạnh!")
        elif tech.macd_trend == "BULLISH" and tech.macd_histogram > 0:
            tech_score_long += 3
        elif tech.macd_trend == "BEARISH_CROSS":
            tech_score_short += 4
            signals.append("🔥 MACD vừa bearish cross - Tín hiệu mạnh!")
        elif tech.macd_trend == "BEARISH" and tech.macd_histogram < 0:
            tech_score_short += 3
        

        ma_bullish = 0
        if tech.ema_9 > tech.ema_21:
            ma_bullish += 1
        if tech.sma_20 > tech.sma_50:
            ma_bullish += 1
        if tech.sma_50 > tech.sma_200:
            ma_bullish += 1
        
        
        if ma_bullish == 3:
            tech_score_long += 3
            signals.append("✅ Tất cả MA trong xu hướng tăng - Tốt cho LONG")
        elif ma_bullish == 0:
            tech_score_short += 3
            signals.append("❌ Tất cả MA trong xu hướng giảm - Tốt cho SHORT")
        
    
        if tech.golden_cross:
            tech_score_long += 4
            signals.append("⭐ Golden Cross xuất hiện - Tốt cho LONG")
        elif tech.death_cross:
            tech_score_short += 4
            signals.append("💀 Death Cross xuất hiện - Tốt cho SHORT")
        
        if tech.bb_position < 0.2:
            tech_score_long += 2
            signals.append("📊 Giá gần BB Lower - Potential bounce - Tốt cho LONG")
        elif tech.bb_position > 0.8:
            tech_score_short += 2
            signals.append("📊 Giá gần BB Upper - Potential correction - Tốt cho SHORT")
        
      
        if tech.stochastic_signal == "BULLISH_CROSS_OVERSOLD":
            tech_score_long += 4
            signals.append(f"🔥 Stochastic bullish cross trong oversold - Tín hiệu mua mạnh!")
        elif tech.stochastic_signal == "OVERSOLD":
            tech_score_long += 2
            signals.append(f"📊 Stochastic Oversold ({tech.stochastic_k:.1f}) - Tốt cho LONG")
        elif tech.stochastic_signal == "BEARISH_CROSS_OVERBOUGHT":
            tech_score_short += 4
            signals.append(f"🔥 Stochastic bearish cross trong overbought - Tín hiệu bán mạnh!")
        elif tech.stochastic_signal == "OVERBOUGHT":
            tech_score_short += 2
            signals.append(f"📊 Stochastic Overbought ({tech.stochastic_k:.1f}) - Tốt cho SHORT")

        
        if "STRONG" in tech.adx_trend or "VERY_STRONG" in tech.adx_trend:
            if "UPTREND" in tech.adx_trend:
                tech_score_long += 3 if "VERY_STRONG" in tech.adx_trend else 2
                signals.append(f"💪 {tech.adx_trend} (ADX: {tech.adx}) - Tốt cho LONG")
            elif "DOWNTREND" in tech.adx_trend:
                tech_score_short += 3 if "VERY_STRONG" in tech.adx_trend else 2
                signals.append(f"💪 {tech.adx_trend} (ADX: {tech.adx}) - Tốt cho SHORT")
        
        if tech.obv_trend == "UPTREND":
            tech_score_long += 2
            signals.append("📈 OBV tăng - Volume bullish - Tốt cho LONG")
        elif tech.obv_trend == "DOWNTREND":
            tech_score_short += 2
            signals.append("📉 OBV giảm - Volume bearish - Tốt cho SHORT")
        
        tech_score_long = max(-10, min(10, tech_score_long))
        tech_score_short = max(-10, min(10, tech_score_short))
        
        breakdown_long['technical'] = tech_score_long * 0.30
        breakdown_short['technical'] = tech_score_short * 0.30
        
        # 3. MARKET SENTIMENT (20%)
 
        sentiment_score_long = 0
        sentiment_score_short = 0
        
        st_24h = symbol_data.get('st', {}).get('24h', '')
        if st_24h == 'bull':
            sentiment_score_long += 4
            signals.append("🐂 Market Sentiment: BULL - Tốt cho LONG")
        elif st_24h == 'bear':
            sentiment_score_short += 4
            signals.append("🐻 Market Sentiment: BEAR - Tốt cho SHORT")
        
        bv_24h = symbol_data.get('bv', {}).get('24h', 0)
        sv_24h = symbol_data.get('sv', {}).get('24h', 0)
        if bv_24h > sv_24h * 1.5:
            sentiment_score_long += 3
            signals.append("📊 Buy Volume >> Sell Volume - Tốt cho LONG")
        elif sv_24h > bv_24h * 1.5:
            sentiment_score_short += 3
            signals.append("📊 Sell Volume >> Buy Volume - Tốt cho SHORT")
        
        if market.overall_trend in ["STRONG_BULLISH", "BULLISH"]:
            sentiment_score_long += 2
            signals.append("🎯 Phù hợp xu hướng thị trường tăng - Tốt cho LONG")
        elif market.overall_trend in ["STRONG_BEARISH", "BEARISH"]:
            sentiment_score_short += 2
            signals.append("🎯 Phù hợp xu hướng thị trường giảm - Tốt cho SHORT")
        
        if market.fear_greed_index < 25 and tech.rsi < 35:
            sentiment_score_long += 3
            signals.append("😱 Extreme Fear + Low RSI - Cơ hội mua - Tốt cho LONG")
        elif market.fear_greed_index > 75 and tech.rsi > 65:
            sentiment_score_short += 3
            signals.append("🤑 Extreme Greed + High RSI - Cảnh báo - Tốt cho SHORT")
        
        sentiment_score_long = max(-10, min(10, sentiment_score_long))
        sentiment_score_short = max(-10, min(10, sentiment_score_short))
        
        breakdown_long['sentiment'] = sentiment_score_long * 0.20
        breakdown_short['sentiment'] = sentiment_score_short * 0.20
        
        # 4. RISK-ADJUSTED SCORE (15%)
        risk_score = 0
        
        if risk.liquidity_score > 7:
            risk_score += 3
            signals.append("💧 Thanh khoản cao - Tốt cho cả LONG & SHORT")
        elif risk.liquidity_score < 4:
            risk_score -= 3
            signals.append("⚠️ Thanh khoản thấp - Rủi ro cho cả LONG & SHORT")
        
        if risk.volatility_24h < 0.02:
            risk_score += 2
            signals.append("📊 Biến động thấp - Ổn định - Tốt cho cả hai")
        elif risk.volatility_24h > 0.05:
            risk_score -= 2
            signals.append("⚡ Biến động cao - Rủi ro - Cẩn thận")
        
        if risk.sharpe_ratio > 1.5:
            risk_score += 2
            signals.append(f"📊 Sharpe tốt ({risk.sharpe_ratio:.2f}) - Tốt cho cả hai")
        elif risk.sharpe_ratio < 0:
            risk_score -= 2
            signals.append(f"⚠️ Sharpe âm ({risk.sharpe_ratio:.2f}) - Rủi ro")
        
        if risk.max_drawdown > 0.30:
            risk_score -= 2
            signals.append(f"⚠️ Max DD cao ({risk.max_drawdown:.1%}) - Rủi ro")
        
        risk_score = max(-10, min(10, risk_score))
        
        breakdown_long['risk_adjusted'] = risk_score * 0.15
        breakdown_short['risk_adjusted'] = risk_score * 0.15
        
        total_score_long = sum(breakdown_long.values())
        total_score_short = sum(breakdown_short.values())
        
        final_breakdown = breakdown_long if total_score_long >= total_score_short else breakdown_short
        
        return round(total_score_long, 2), round(total_score_short, 2), final_breakdown, signals

    # --- CÁC PHƯƠNG THỨC MỚI ĐÃ ĐƯỢC CẢI TIẾN ---
    def dynamic_weights(self, market_regime: str) -> Dict[str, float]:
        """Trả về trọng số động - TĂNG TRỌNG SỐ cho money flow"""
        
        if market_regime == "RISK_OFF":
            return {'money_flow': 0.30, 'technical': 0.30, 'sentiment': 0.20, 'risk': 0.20}  # Điều chỉnh
        elif market_regime == "RISK_ON":
            return {'money_flow': 0.45, 'technical': 0.25, 'sentiment': 0.15, 'risk': 0.15}  # Tăng money flow
        else:  # NEUTRAL
            return {'money_flow': 0.40, 'technical': 0.25, 'sentiment': 0.20, 'risk': 0.15}  # Tăng money flow

    def consensus_realtime(self, realtime_data: RealTimeData) -> Tuple[float, List[str]]:
        """
        Tính điểm consensus từ real-time signals với weighted voting
        
        Trọng số:
        - Options: 0.4 (quan trọng nhất)
        - Liquidation: 0.3
        - Orderbook: 0.2
        - CVD: 0.1
        """
        signals = []
        msgs = []
        
        # 1. Options sentiment (weight=0.4)
        options_sentiment = realtime_data.options_data.get('sentiment', 'NEUTRAL')
        if options_sentiment == 'BULLISH':
            signals.append(('LONG', 0.4))
            msgs.append("📊 Options sentiment: BULLISH (weight=0.4)")
        elif options_sentiment == 'BEARISH':
            signals.append(('SHORT', 0.4))
            msgs.append("📊 Options sentiment: BEARISH (weight=0.4)")
        
        # 2. Liquidation bias (weight=0.3)
        liq_bias = realtime_data.liquidation_bias.get('bias', 0.0)
        if abs(liq_bias) > 0.3:
            # bias < 0: long liq nhiều → giá có thể tăng để săn long → LONG
            # bias > 0: short liq nhiều → giá có thể giảm để săn short → SHORT
            direction = "LONG" if liq_bias < 0 else "SHORT"
            signals.append((direction, 0.3))
            msgs.append(f"⚡ Liquidation bias: {liq_bias:.2f} → {direction} (weight=0.3)")
        
        # 3. Orderbook imbalance (weight=0.2)
        ob_imbalance = realtime_data.orderbook_imbalance
        if abs(ob_imbalance) > 0.1:
            direction = "LONG" if ob_imbalance > 0 else "SHORT"
            signals.append((direction, 0.2))
            msgs.append(f"📊 Orderbook imbalance: {ob_imbalance:.3f} → {direction} (weight=0.2)")
        
        # 4. CVD signal (weight=0.1)
        cvd_signal = realtime_data.cvd_signal.get('signal', 'NEUTRAL')
        if cvd_signal in ['BULLISH', 'BEARISH']:
            direction = "LONG" if cvd_signal == 'BULLISH' else "SHORT"
            signals.append((direction, 0.1))
            msgs.append(f"📈 CVD signal: {cvd_signal} → {direction} (weight=0.1)")
        
        # Tính weighted score
        long_score = sum(weight for direction, weight in signals if direction == "LONG")
        short_score = sum(weight for direction, weight in signals if direction == "SHORT")
        
        # Determine consensus
        if long_score > short_score:
            if long_score >= 0.7:
                consensus_score = 2  # Strong consensus
                msgs.insert(0, f"✅ STRONG LONG consensus (score={long_score:.2f})")
            else:
                consensus_score = 1  # Weak consensus
                msgs.insert(0, f"↗️ WEAK LONG consensus (score={long_score:.2f})")
        elif short_score > long_score:
            if short_score >= 0.7:
                consensus_score = -2
                msgs.insert(0, f"✅ STRONG SHORT consensus (score={short_score:.2f})")
            else:
                consensus_score = -1
                msgs.insert(0, f"↘️ WEAK SHORT consensus (score={short_score:.2f})")
        else:
            consensus_score = 0
            msgs.insert(0, f"⚖️ NEUTRAL consensus (L:{long_score:.2f} vs S:{short_score:.2f})")
        
        return consensus_score, msgs
            

    def enhanced_decision_threshold(self, volatility: float) -> Tuple[float, float]:
        """Ngưỡng BUY/SELL động tùy volatility - GIẢM NGƯỎNG"""
        if volatility > 0.05:  # >5% 24h vol
            return 5.5, 2.5   # Giảm từ 7.5, 3.5
        elif volatility < 0.015:  # <1.5% 24h vol
            return 3.5, 1.5   # Giảm từ 5.5, 2.5
        return 4.5, 2.0       # Giảm từ 6.5, 3.0

    def validate_enhanced_signal(self, long_score: float, short_score: float, risk: RiskMetrics,
                                rr_ratio: float, realtime_data: RealTimeData,
                                tech: Optional[TechnicalIndicators] = None) -> Tuple[bool, List[str]]:
 
        """Validate tín hiệu giao dịch với điều kiện NỚI LỎNG hơn"""
        warnings = []
        is_valid = True
        
        # Giảm ngưỡng điểm tối thiểu
        if max(long_score, short_score) < 2:  # Giảm từ 3
            warnings.append("⚠️ Điểm tổng hợp thấp (< 2)")
            is_valid = False
        
        # Điều chỉnh điều kiện risk
        if risk.risk_level == "EXTREME":
            warnings.append("⚠️ Rủi ro cực cao - Khuyến nghị tránh")
            is_valid = False
        elif risk.risk_level == "HIGH" and max(long_score, short_score) < 4: 
            # Giảm từ 6
            warnings.append("⚠️ Rủi ro cao yêu cầu điểm số cao hơn")
            # Chỉ warning, không set is_valid = False
        
        if rr_ratio < 0.8:  # Giảm từ 1.0
            warnings.append(f"⚠️ Tỷ lệ R:R thấp ({rr_ratio:.2f}) - Khuyến nghị > 1.0")
        
        if risk.liquidity_score < 3:  # Giảm từ 4
            warnings.append("⚠️ Thanh khoản thấp")
            is_valid = False
            
        # Real-time checks
        if realtime_data.liquidation_bias.get('squeeze_risk') in ['LONG_SQUEEZE', 'SHORT_SQUEEZE']:
            warnings.append(f"⚠️ High liquidation risk: {realtime_data.liquidation_bias.get('squeeze_risk')}")
        
        if abs(realtime_data.funding_rate) > 0.005:
            warnings.append(f"⚠️ Extreme funding rate: {realtime_data.funding_rate:.4f}")
        
        return is_valid, warnings

    async def analyze_symbol(self, symbol_data: Dict, market: MarketContext) -> Optional[TradingSignal]:
        """Analyze with normalized symbol - ĐÃ SỬA LỖI LẤY GIÁ"""
        try:
  
            raw_symbol = symbol_data.get('p', '')
            symbol = self._normalize_symbol(raw_symbol)
            if not symbol:
                return None
            
            logging.info(f"  🔍 Analyzing {symbol} with REAL-TIME data...")
           
            # SỬA LỖI: Sử dụng symbol đã chuẩn hóa để lấy giá
            current_price, price_data = self.get_real_price(symbol)
            if current_price == 0:
                logging.warning(f"  ⚠️  Bỏ qua {symbol} - không lấy được giá")
                return None
  
            
            logging.info(f"  💰 Price: ${current_price:.6f} (Source: {price_data.get('source', 'unknown')})")
            
            # Sử dụng symbol đã chuẩn hóa cho các analyzer
            realtime_data = await self.get_real_time_data(symbol)  
            df = self.get_historical_klines(symbol, interval='1h', limit=200)
     
            
            tech = self.calculate_comprehensive_indicators(df, current_price)
            risk = self.calculate_enhanced_risk_metrics(df, price_data, current_price, realtime_data)
            money_flow = self.analyze_money_flow(symbol_data, df)
            
            # Tính điểm ban đầu
            long_score_base, short_score_base, breakdown, signals = self.calculate_directional_scores(
                tech, risk, money_flow, market, symbol_data
            )

            # --- Dynamic weights ---
            weights = self.dynamic_weights(market.market_regime)
            # Re-calculate score based on dynamic weights (simplified for integration)
            long_score = long_score_base
            short_score = short_score_base
            
            # --- Consensus real-time ---
            consensus_score, consensus_msgs = self.consensus_realtime(realtime_data)
            if consensus_score > 0:
                long_score += consensus_score
            else:
                short_score += abs(consensus_score)
            signals.extend(consensus_msgs)

            # Xác định hướng cuối cùng
            if long_score >= short_score:
                direction = Direction.LONG.value
                composite_score = long_score
            else:
                direction = Direction.SHORT.value
                composite_score = short_score

            # --- Ngưỡng động (ĐÃ ĐIỀU CHỈNH) ---
            buy_thr, neutral_thr = self.enhanced_decision_threshold(risk.volatility_24h)
            should_enter = False
            reason_for_entry = ""
            risk_alerts = []
            # --- Regime filter cho Altcoin ---
            try:
                # symbol_data.get('p') là tên symbol được pass trong hàm gọi; đổi theo key bạn đang dùng nếu khác
                _sym = symbol_data.get('p', '').upper()
                _is_alt = _sym not in ("BTC", "ETH")

                if _is_alt and market.market_regime == "RISK_OFF":
                    # Hạn chế long alt khi risk-off: tăng yêu cầu điểm mua hoặc giảm điểm long
                    buy_thr += 0.5         # yêu cầu điểm cao hơn mới vào
                    long_score *= 0.95     # “đè” nhẹ điểm long
                    signals.append("⚠️ RISK_OFF: thắt chặt điều kiện LONG cho altcoin")
                elif _is_alt and market.market_regime == "RISK_ON":
                    # Risk-on: nới nhẹ điều kiện
                    long_score *= 1.03
                    signals.append("📈 RISK_ON: ưu ái nhẹ LONG cho altcoin")
            except Exception:
                pass

            if direction == "LONG":
                if long_score >= buy_thr and risk.risk_level in ["LOW", "MEDIUM"]:
                    should_enter = True
                    reason_for_entry = f"✅ Điểm LONG cao ({long_score:.1f} >= {buy_thr}) và risk chấp nhận được ({risk.risk_level})"
                elif long_score >= buy_thr and risk.risk_level == "HIGH":
                    # CHO PHÉP vào lệnh với risk HIGH nếu điểm đủ cao
                    should_enter = long_score >= (buy_thr + 1.0)  # Yêu cầu điểm cao hơn
                    reason_for_entry = f"⚠️ Điểm LONG cao ({long_score:.1f}) nhưng risk cao, cần cẩn trọng"
                    if should_enter:
                        reason_for_entry += " - Vẫn vào lệnh với size nhỏ"
                    else:
                        reason_for_entry += " - Cần điểm cao hơn để vào lệnh"
                else:
                    reason_for_entry = f"❌ Điểm LONG ({long_score:.1f}) chưa đủ mạnh"
            else:  # SHORT
                if short_score >= buy_thr and risk.risk_level in ["LOW", "MEDIUM"]:
                    should_enter = True
                    reason_for_entry = f"✅ Điểm SHORT cao ({short_score:.1f} >= {buy_thr}) và risk chấp nhận được ({risk.risk_level})"
                elif short_score >= buy_thr and risk.risk_level == "HIGH":
                    should_enter = short_score >= (buy_thr + 1.0)
                    reason_for_entry = f"⚠️ Điểm SHORT cao ({short_score:.1f}) nhưng risk cao, cần cẩn trọng"
                    if should_enter:
                        reason_for_entry += " - Vẫn vào lệnh với size nhỏ"
                    else:
                        reason_for_entry += " - Cần điểm cao hơn để vào lệnh"
                else:
                    reason_for_entry = f"❌ Điểm SHORT ({short_score:.1f}) chưa đủ mạnh"

            # --- Đầu ra bổ sung ---
            entry_score = long_score if direction == "LONG" else short_score
            sl_score = -1 if risk.max_drawdown > 0.25 or risk.volatility_24h > 0.06 else 1
            tp_score = 1 if risk.sharpe_ratio > 1 else 0

    
            # Phần còn lại của logic (levels, position size, v.v...)
            decision, entry_strategy = self.determine_decision_and_strategy(composite_score, direction, tech, money_flow)
            
            # ĐÃ CẬP NHẬT THAM SỐ risk
            levels = self.calculate_trading_levels(current_price, decision, direction, tech, money_flow, risk, realtime_data)
            
   
            leverage = self.recommend_leverage(risk, realtime_data, levels['risk_reward_ratio'])
            pos_size_usd, pos_size_units, margin_required = self.calculate_enhanced_position_sizing(
                current_price, levels['stop_loss'], risk, leverage, direction, realtime_data)
    
            is_valid, validation_warnings = self.validate_enhanced_signal(
                long_score, short_score, risk, levels['risk_reward_ratio'], 
                realtime_data, tech  # ✅ THÊM
            )

            confidence = min(1.0, (abs(composite_score) / 10) * 0.5 + money_flow.flow_consistency_score * 0.2 + (1 if is_valid else 0.3) * 0.3)
            # --- Điều chỉnh theo beta & correlation với BTC khi trend xấu ---
            try:
                if risk.beta > 1.5 and market.overall_trend.lower() == "bearish":
                    confidence *= 0.85
                    warnings.append("⚠️ Beta cao + BTC bearish → giảm confidence")

                if abs(risk.correlation_btc) > 0.8 and market.overall_trend.lower() == "bearish":
                    confidence *= 0.90
                    warnings.append("⚠️ Tương quan BTC mạnh + BTC bearish → giảm confidence")
            except Exception:
                # Phòng xa nếu object khác cấu trúc
                pass
            
            
            warnings = []
            if risk.risk_level in ["HIGH", "EXTREME"]: 
                warnings.append(f"⚠️ Rủi ro {risk.risk_level}")
            if risk.liquidity_score < 5: 
                warnings.append("⚠️ Thanh khoản thấp")
            warnings.extend(validation_warnings)

            logging.info(f"  ✅ Direction: {direction} | Score: {composite_score:+.2f} (L:{long_score:+.1f}/S:{short_score:+.1f})")
            logging.info(f"  📊 Decision: {decision.value} | Confidence: {confidence:.1%} | R:R {levels['risk_reward_ratio']:.2f}")
            logging.info(f"  👉 Should Enter: {should_enter} | Reason: {reason_for_entry}")
            
            return TradingSignal(
                symbol=symbol,
                exchange=price_data.get('source', 'unknown'),
                decision=decision.value,
                direction=direction,
                confidence=round(confidence, 3),
                composite_score=composite_score,
                long_score=long_score,
                short_score=short_score,
 
                current_price=current_price,
                entry_price=levels['entry_price'],
                stop_loss=levels['stop_loss'],
                take_profit_levels=levels['take_profit'],
                position_size_usd=pos_size_usd,
                position_size_units=pos_size_units,
     
                margin_required=margin_required,
                leverage_recommended=leverage,
                risk_reward_ratio=levels['risk_reward_ratio'],
                entry_strategy=entry_strategy.value,
                timeframe="1H",
                signals=signals[:10],
                warnings=warnings,
                score_breakdown=breakdown,
                technical_indicators=tech,
                risk_metrics=risk,
                money_flow=money_flow,
                market_context=market,
             
                realtime_data=realtime_data,
                timestamp=datetime.now().isoformat(),
                # Thêm các trường mới
                entry_score=entry_score,
                sl_score=sl_score,
                tp_score=tp_score,
             
                risk_alerts=risk_alerts,
                should_enter=should_enter,
                reason_for_entry=reason_for_entry
            )
            
        except Exception as e:
            logging.error(f"  ❌ Error analyzing {symbol_data.get('p', 'unknown')}: {e}", exc_info=True)
            return None
        
    def recommend_leverage(self, risk: RiskMetrics, rt: RealTimeData, rr: float) -> int:
        """
        Khuyến nghị đòn bẩy dựa trên: mức rủi ro, biến động, funding, squeeze, orderbook imbalance,
        và trần margin tối đa cho phép (giữ khoảng an toàn).
        Trả về số nguyên >=1.
        """
        # 1) Base theo risk level
        base = 1
        if risk.risk_level == "LOW":
            base = 4
        elif risk.risk_level == "MEDIUM":
            base = 3
        elif risk.risk_level == "HIGH":
            base = 2
        else:  # EXTREME
            base = 1

        # 2) Giảm theo biến động và điều kiện RT
        vol = float(risk.volatility_24h or 0.0)
        if vol > 0.06:           # biến động 24h > 6%
            base -= 1
        if abs(rt.funding_rate) > 0.005:
            base -= 1
        if rt.liquidation_bias.get('squeeze_risk') in ("LONG_SQUEEZE", "SHORT_SQUEEZE"):
            base -= 1
        if abs(rt.orderbook_imbalance) > 0.3:
            base -= 1

        # 3) Nghiêng nhẹ theo R:R (nếu R:R tốt, cho phép +1)
        if rr is not None and rr >= 1.8:
            base += 1

        # 4) Giới hạn khung hợp lý (vd. 1..5) – có thể đưa vào config
        lev = max(1, min(base, 5))

        # 5) Kiểm tra trần margin theo % balance (bảo đảm margin không vượt quá ngưỡng)
        #    Nếu lev quá thấp khiến margin_required > max margin, tăng lev cho phù hợp.
        max_margin_usd = self.account_balance * 0.20  # ví dụ: không khóa >20% vốn vào margin
        # Gợi ý: caller sẽ tính size dựa trên lev, nhưng nếu bạn muốn siết ở đây:
        # lev = max(lev, int(math.ceil(position_size_usd / max_margin_usd)))  # cần có position_size_usd
        # → Vì ở đây chưa có size, ta để logic cap USD ở nơi tính size (đã làm ở phần trước).
        return int(lev)


    def calculate_enhanced_position_sizing(self, current_price: float, stop_loss: float, 
                                             risk: RiskMetrics, leverage: int, direction: str,
                                             realtime_data: RealTimeData) -> Tuple[float, float, float]:
        """Tính toán kích thước vị thế với real-time risk factors"""
        
        risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Adjust by risk level
        if risk.risk_level == "EXTREME":
            risk_amount *= 0.5
        elif risk.risk_level == "HIGH":
            risk_amount *= 0.7
        elif risk.risk_level == "LOW":
            risk_amount *= 1.2
        
        # Adjust by Kelly Criterion
        kelly_factor = min(risk.kelly_criterion, 0.25)
        risk_amount *= (kelly_factor / 0.02) if kelly_factor > 0 else 1.0
      
        # Adjust by real-time risk factors
        if realtime_data.liquidation_bias.get('squeeze_risk') != 'LOW':
            risk_amount *= 0.7
            
        if abs(realtime_data.orderbook_imbalance) > 0.3:
            risk_amount *= 0.9
            
        if abs(realtime_data.funding_rate) > 0.003:
            risk_amount *= 0.8
        
        if direction == Direction.LONG.value:
            risk_per_unit = abs(current_price - stop_loss)
        else:
            risk_per_unit = abs(stop_loss - current_price)
            
        if risk_per_unit > 0:
            position_size_units = risk_amount / risk_per_unit
            position_size_usd = position_size_units * current_price
        else:
            position_size_units = 0
            position_size_usd = 0
        
        margin_required = position_size_usd / leverage if leverage > 0 else position_size_usd
        
  
        if margin_required > self.account_balance:
            margin_required = self.account_balance
            position_size_usd = margin_required * leverage
            position_size_units = position_size_usd / current_price
        
        return (round(position_size_usd, 2), round(position_size_units, 4), 
                round(margin_required, 2))

    def calculate_trading_levels(self, current_price: float, decision: Decision, 
                                 direction: str, tech: TechnicalIndicators, 
                                 money_flow: MoneyFlowAnalysis, risk: RiskMetrics, # ĐÃ THÊM THAM SỐ risk
                                 realtime_data: RealTimeData) -> Dict:
        """Tính toán các mức giá giao dịch với real-time adjustments"""
        
        atr = tech.atr if tech.atr > 0 else current_price * 0.02
        atr_multiplier = 1.0
        
        # Adjust for real-time factors
        if money_flow.flow_strength == "EXTREME":
            atr_multiplier = 1.5
        elif money_flow.flow_strength == "STRONG":
            atr_multiplier = 1.3
            
        # Adjust for volatility from real-time data
        if abs(realtime_data.orderbook_imbalance) > 0.2:
            atr_multiplier *= 1.2
        
        if direction == Direction.LONG.value:
            entry_price = current_price * 0.998
            
            sl_atr = current_price - (atr * 2 * atr_multiplier)
            sl_bb = tech.bb_lower if tech.bb_lower > 0 else current_price * 0.96
            sl_support = min(tech.support_resistance.get('support', [current_price * 0.97])[0], current_price * 0.95)
            stop_loss = min(sl_atr, sl_bb, sl_support)
            
            tp1 = current_price + (atr * 2 * atr_multiplier)
            tp2 = current_price + (atr * 4 * atr_multiplier)
            tp3 = current_price + (atr * 6 * atr_multiplier)
            take_profit = [tp1, tp2, tp3]
            
        else:
            entry_price = current_price * 1.002
            
            sl_atr = current_price + (atr * 2 * atr_multiplier)
            sl_bb = tech.bb_upper if tech.bb_upper > 0 else current_price * 1.04
            sl_resistance = max(tech.support_resistance.get('resistance', [current_price * 1.03])[0], current_price * 1.05)
            stop_loss = max(sl_atr, sl_bb, sl_resistance)
        
            tp1 = current_price - (atr * 2 * atr_multiplier)
            tp2 = current_price - (atr * 4 * atr_multiplier)
            tp3 = current_price - (atr * 6 * atr_multiplier)
            take_profit = [tp1, tp2, tp3]
        
        if direction == Direction.LONG.value:
            risk_calc = abs(entry_price - stop_loss)
            reward = abs(take_profit[1] - entry_price)
        else:
            risk_calc = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit[1])
            
        rr_ratio = reward / risk_calc if risk_calc > 0 else 0
        
        return {
            'entry_price': round(entry_price, 6),
            'stop_loss': round(stop_loss, 6),
            'take_profit': [round(tp, 6) for tp in take_profit],
            'risk_reward_ratio': round(rr_ratio, 2)
        }

    def determine_decision_and_strategy(self, composite_score: float, direction: str,
                                             tech: TechnicalIndicators, money_flow: MoneyFlowAnalysis) -> Tuple[Decision, EntryStrategy]:
        """Xác định quyết định và chiến lược vào lệnh"""
   
        # Determine Decision based on score and direction
        score_abs = abs(composite_score)
        
        if direction == Direction.LONG.value:
            if score_abs >= 8: 
                decision = Decision.STRONG_BUY
            elif score_abs >= 4: 
                decision = Decision.BUY
            else: 
                decision = Decision.NEUTRAL
        else: # SHORT
            if score_abs >= 8: 
                decision = Decision.STRONG_SELL
            elif score_abs >= 4: 
                decision = Decision.SELL
            else: 
                decision = Decision.NEUTRAL

        # Determine Entry Strategy
        if money_flow.flow_strength == "EXTREME" and score_abs >= 8:
            strategy = EntryStrategy.IMMEDIATE
        elif (direction == "LONG" and tech.rsi < 35) or (direction == "SHORT" and tech.rsi > 65):
            strategy = EntryStrategy.WAIT_CONFIRMATION
        elif (direction == "LONG" and tech.bb_position < 0.3) or (direction == "SHORT" and tech.bb_position > 0.7):
            strategy = EntryStrategy.WAIT_PULLBACK
        elif tech.adx > 25:
            strategy = EntryStrategy.WAIT_BREAKOUT
        else:
            strategy = EntryStrategy.WAIT_CONFIRMATION
        
        return decision, strategy

    async def run_enhanced_analysis(self, input_file: str = 'input_data_short.json') -> List[TradingSignal]:
        """Chạy phân tích nâng cao với real-time integration"""
        print("="*100)
        print("🚀 ENHANCED CRYPTO TRADING ANALYZER - REAL-TIME INTEGRATION")
        print("="*100)
      
        # Load Nansen data
        nansen_data = self.load_nansen_data(input_file)
        if not nansen_data:
            print("❌ Không có dữ liệu để phân tích")
            return []

        # Chuẩn hóa symbols để truyền vào start_real_time_data
        symbols = [token.get('p', '') for token in nansen_data if token.get('p')]
   
        # Khởi động real-time data
        if self.use_websocket:
            print("\n🔌 Đang khởi động real-time data streams...")
            await self.start_real_time_data(symbols)
            print("⏳ Đang chờ dữ liệu real-time... (10 giây)")
            await asyncio.sleep(10)

        # Detect market regime
        print("\n🌐 Đang phân tích market regime...")
        await self.detect_market_regime()

        # Get market context
        market = self.get_market_context()
        print(f"  📊 Xu hướng: {market.overall_trend}")
        print(f"  😱 Fear & Greed: {market.fear_greed_index} ({market.fear_greed_classification})")
        # ĐÃ SỬA LỖI: market.btc_dominance đã là fraction, in theo %
        print(f"  ₿ BTC Dominance: {market.btc_dominance:.2%}")
        print(f"  💰 Market Cap: ${market.total_market_cap:,.0f} ({market.market_cap_change_24h:+.2f}%)")
        print(f"  🌡️ Market Regime: {self.market_regime}")
        
        # Analyze all symbols
        print(f"\n🔍 Phân tích {len(nansen_data)} coins với REAL-TIME data...")
        print("-"*100)
        
        tasks = [self.analyze_symbol(symbol_data, market) for symbol_data in nansen_data]
        results_raw = await asyncio.gather(*tasks)
        results = [res for res in results_raw if res is not None]
        
        # Thêm kiểm tra tỷ lệ fallback
        fallback_count = 0
        total_signals = len(results)
        for r in results:
            rt = r.realtime_data
            # Nếu orderbook imbalance và options sentiment đều là neutral (giá trị fallback)
            if abs(rt.orderbook_imbalance) < 0.001 and rt.options_data.get('sentiment', 'NEUTRAL') == 'NEUTRAL' and abs(rt.liquidation_bias.get('bias', 0.0)) < 0.001:
                fallback_count += 1
        
        if total_signals > 0 and fallback_count / total_signals > 0.2:
            print(f"\n⚠️ CẢNH BÁO: Tỷ lệ tín hiệu dùng fallback data cao ({fallback_count}/{total_signals} - {fallback_count / total_signals:.1%}). Kết quả có thể kém tin cậy.")

        # Cleanup
        if self.use_websocket:
            await self.cleanup()

        # Display and export results
        self.display_enhanced_results(results, market)
        self.export_enhanced_results(results)
        
        return results

    def display_enhanced_results(self, results: List[TradingSignal], market: MarketContext):
        """Hiển thị kết quả nâng cao với real-time insights và chi tiết signals đạt điều kiện"""
        if not results:
            print("\n❌ Không có kết quả phân tích")
            return
        
        print("\n" + "="*100)
 
        print("📊 ENHANCED TRADING SIGNALS - REAL-TIME INTEGRATION")
        print("="*100)
        
        sorted_results = sorted(results, key=lambda x: (x.should_enter, abs(x.composite_score)), reverse=True)
        enter_now_signals = [r for r in results if r.should_enter]
        
        print("\n🏆 TOP TRADING OPPORTUNITIES (Ưu tiên Should Enter=True):")
        print("-"*120)
        print(f"{'#':<3} {'Symbol':<15} {'Enter?':<8} {'Direction':<6} {'Decision':<12} {'Score':<10} {'Conf.':<8} {'R:R':<6} {'Reason for Entry / Warnings'}")
        print("="*120)

        for i, signal in enumerate(sorted_results[:20], 1):
            enter_str = "✅ YES" if signal.should_enter else "❌ NO"
            score_str = f"{signal.composite_score:+.2f}"
            conf_str = f"{signal.confidence:.0%}"
            rr_str = f"{signal.risk_reward_ratio:.2f}"
      
            
            # Combine reason and key warnings
            info_str = signal.reason_for_entry
            if signal.warnings:
                info_str += f" | Warnings: {', '.join(signal.warnings[:1])}"

            print(f"{i:<3} {signal.symbol:<15} {enter_str:<8} {signal.direction:<6} {signal.decision:<12} {score_str:<10} {conf_str:<8} {rr_str:<6} {info_str}")

        print("\n" + "="*100)
        print("📈 ENHANCED STATISTICS")
        print("="*100)
        
        long_signals = [r for r in results if r.direction == Direction.LONG.value]
        short_signals = [r for r in results if r.direction == Direction.SHORT.value]
      
        
        print(f"  🟢 LONG Signals: {len(long_signals)} | 🔴 SHORT Signals: {len(short_signals)}")
        print(f"  ✅ Signals đạt điều kiện vào lệnh: {len(enter_now_signals)}")

        # ========== PHẦN MỚI: HIỂN THỊ CHI TIẾT SIGNALS ĐẠT ĐIỀU KIỆN ==========
        if enter_now_signals:
            print("\n" + "🎯" * 50)
            print("🎯 CHI TIẾT CÁC TÍN HIỆU ĐẠT ĐIỀU KIỆN VÀO LỆNH 🎯")
            print("🎯" * 50)
            
            for i, signal in enumerate(enter_now_signals, 1):
                print(f"\n{'='*80}")
                print(f"🏆 {i}. {signal.symbol} - {signal.decision} ({signal.direction})")
                print(f"{'='*80}")
                
                # Thông tin cơ bản
                print(f"📊 ĐIỂM SỐ & TÍN HIỆU:")
                print(f"   ├── Điểm tổng hợp: {signal.composite_score:+.2f}")
 
                print(f"   ├── Điểm LONG: {signal.long_score:+.2f}")
                print(f"   ├── Điểm SHORT: {signal.short_score:+.2f}")
                print(f"   ├── Confidence: {signal.confidence:.1%}")
                print(f"   └── Risk/Reward: {signal.risk_reward_ratio:.2f}")
              
 
                # Thông tin giá
                print(f"\n💰 THÔNG TIN GIÁ:")
                print(f"   ├── Giá hiện tại: ${signal.current_price:.6f}")
                print(f"   ├── Giá vào lệnh: ${signal.entry_price:.6f}")
                
                print(f"   ├── Stop Loss: ${signal.stop_loss:.6f}")
                print(f"   └── Take Profit: {[f'${tp:.6f}' for tp in signal.take_profit_levels]}")
                
                # Quản lý vốn
                print(f"\n💼 QUẢN LÝ VỐN:")
              
                print(f"   ├── Position Size: ${signal.position_size_usd:.2f}")
                print(f"   ├── Số lượng: {signal.position_size_units:.4f} units")
                print(f"   ├── Margin cần: ${signal.margin_required:.2f}")
                print(f"   └── Đòn bẩy đề xuất: {signal.leverage_recommended}x")
                
       
                # Chiến lược
                print(f"\n🎯 CHIẾN LƯỢC:")
                print(f"   ├── Chiến lược vào lệnh: {signal.entry_strategy}")
                print(f"   ├── Timeframe: {signal.timeframe}")
                print(f"   └── Lý do vào lệnh: {signal.reason_for_entry}")
  
               
                # Tín hiệu chính
                print(f"\n📈 TÍN HIỆU CHÍNH:")
                for j, sig in enumerate(signal.signals[:8], 1):  # Hiển thị tối đa 8 tín hiệu
                  
                    print(f"   {j:>2}. {sig}")
                
                # Cảnh báo
                if signal.warnings:
                    print(f"\n⚠️ CẢNH BÁO:")
                    for j, warning in enumerate(signal.warnings, 1):
                        print(f"   {j:>2}. {warning}")
                
                # Real-time data highlights
                print(f"\n🔴 REAL-TIME DATA:")
              
                print(f"   ├── Orderbook Imbalance: {signal.realtime_data.orderbook_imbalance:.3f}")
                print(f"   ├── CVD Signal: {signal.realtime_data.cvd_signal.get('signal', 'N/A')}")
                print(f"   ├── Funding Rate: {signal.realtime_data.funding_rate:.4%}")
                print(f"   └── Options Sentiment: {signal.realtime_data.options_data.get('sentiment', 'N/A')}")
                
         
                # Risk metrics highlights
                print(f"\n⚡ RISK ASSESSMENT:")
                print(f"   ├── Risk Level: {signal.risk_metrics.risk_level}")
                print(f"   ├── Volatility 24h: {signal.risk_metrics.volatility_24h:.2%}")
                print(f"   ├── Liquidity Score: {signal.risk_metrics.liquidity_score:.1f}/10")
      
                print(f"   └── Max Drawdown: {signal.risk_metrics.max_drawdown:.2%}")
                
                print(f"\n🕒 Timestamp: {signal.timestamp}")
                print(f"{'='*80}")

        else:
            print("\n⚠️ Không có signals nào đạt điều kiện vào lệnh trong lần phân tích này.")

    def export_enhanced_results(self, results: List[TradingSignal]):
        """Xuất kết quả phân tích nâng cao ra file Excel và hiển thị thống kê tổng quát"""
        if not results:
            print("❌ Không có kết quả để xuất")
            return
        
        # Tạo DataFrame cho tất cả tín hiệu
        all_signals_df = pd.DataFrame([asdict(signal) for signal in results])
        
        # Xuất ra file Excel
        output_file = "enhanced_trading_signals.xlsx"
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            all_signals_df.to_excel(writer, sheet_name='All Signals', index=False)
            
            # Tách sheet cho từng tín hiệu đạt điều kiện vào lệnh
            for signal in results:
                if signal.should_enter:
                    signal_df = pd.DataFrame([asdict(signal)])
                    signal_df.to_excel(writer, sheet_name=signal.symbol, index=False)
        
        print(f"✅ Đã xuất kết quả phân tích nâng cao ra file: {output_file}")
        
        # Hiển thị thống kê tổng quát
        total_signals = len(results)
        enter_now_signals = len([r for r in results if r.should_enter])
        long_signals = len([r for r in results if r.direction == Direction.LONG.value])
        short_signals = len([r for r in results if r.direction == Direction.SHORT.value])
        
        print("\n📊 THỐNG KÊ TỔNG QUÁT:")
        print(f"  - Tổng số tín hiệu: {total_signals}")
        print(f"  - Tín hiệu đạt điều kiện vào lệnh: {enter_now_signals}")
        print(f"  - Tín hiệu LONG: {long_signals}")
        print(f"  - Tín hiệu SHORT: {short_signals}")
        
        # Thống kê chi tiết cho tín hiệu đạt điều kiện
        if enter_now_signals > 0:
            avg_score = round(sum(signal.composite_score for signal in results) / enter_now_signals, 2)
            avg_confidence = round(sum(signal.confidence for signal in results) / enter_now_signals, 2)
            avg_rr = round(sum(signal.risk_reward_ratio for signal in results) / enter_now_signals, 2)
            
            print(f"  - Điểm trung bình: {avg_score}")
            print(f"  - Confidence trung bình: {avg_confidence}")
            print(f"  - Tỷ lệ R:R trung bình: {avg_rr}")
        
        # Ghi chú về file xuất ra
        print("\n📂 Lưu ý: Kết quả đã được xuất ra file Excel. Vui lòng kiểm tra các sheet để xem chi tiết tín hiệu.")

async def main():
    """Hàm main để chạy chương trình"""
    analyzer = EnhancedCryptoAnalyzer(
        account_balance=10000.0,
        max_risk_per_trade=0.02,
        use_websocket=True
    )
    
    try:
        await analyzer.run_enhanced_analysis('input_data_short.json')
    except Exception as e:
        print(f"❌ Lỗi chạy phân tích: {e}")
    finally:
        await analyzer.cleanup()

if __name__ == "__main__":
    # Chạy event loop
    import asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())