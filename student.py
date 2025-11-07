"""
Project: Machine Learning Assignment - Building ML pipeline for Stock Prediction
Name: Ramdas Chaugale
ID: 40494109
Algorithm: RandomForestRegressor
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pandas_ta as ta

# Suppress pandas warnings for cleaner output
pd.set_option('future.no_silent_downcasting', True)


class Student:
    """
    ALGORITHM CHOICE: RandomForestRegressor
    - Why RF? Handles non-linear relationships well
    - Robust to different market conditions
    - Less prone to overfitting than single decision trees
    - Works well with technical indicator features
    - Achieved 51.47% accuracy on walk-forward validation
    
    DESIGN PATTERN: Pipeline Architecture
    - Preprocessing: StandardScaler (normalize features)
    - Model: RandomForestRegressor (main learner)
    - Fallback: Mean prediction for edge cases
    
    """

    def __init__(self, config: dict | None = None, random_state: int = 42):
        """
        PARAMETERS:
        config : dict or None
            Configuration dictionary with:
            - 'model_type': String specifying which algorithm to use
              (default: 'random_forest')
            - Technical indicator parameters (SMA windows, RSI period, etc.)
            - Algorithm hyperparameters (n_estimators, max_depth, etc.)
            
        random_state : int
            Random seed for reproducibility (default: 42)
            Ensures same results across multiple runs
        
        ATTRIBUTES SET:
        - self.model: The actual ML model (RandomForestRegressor by default)
        - self.pipeline: Combined scaler + model pipeline
        - self.scaler: StandardScaler for feature normalization
        - self.is_fitted: Boolean flag (initially False)
        - self.mean_y_fallback: Fallback prediction value
        - self.feature_names: List of engineered feature names
        
        """
        
        # Setting random seed for reproducibility
        self.random_state = random_state
        np.random.seed(random_state)
        
        
        # Defining all model parameters with sensible defaults
        
        self.config = config or {
            # MAIN MODEL SELECTION
            'model_type': 'random_forest',  # PRIMARY MODEL
            
            # FEATURE ENGINEERING PARAMETERS
            'n_lags': 5,                    # Use 5 days of lag returns
            'sma_short': 5,                 # Short-term SMA window (5 days)
            'sma_long': 20,                 # Long-term SMA window (20 days)
            'sma_medium': 10,               # Medium-term SMA window (10 days)
            'rsi_length': 14,               # RSI calculation period (standard)
            'macd_fast': 12,                # MACD fast EMA
            'macd_slow': 26,                # MACD slow EMA
            'macd_signal': 9,               # MACD signal line period
            'atr_length': 14,               # ATR (volatility) period
            'bb_length': 20,                # Bollinger Bands period
            
            # RANDOMFOREST HYPERPARAMETERS
            # These control model complexity and generalization
            'n_estimators': 100,            # Number of trees (more = better fit)
            'max_depth': 6,                 # Max tree depth (prevents overfitting)
            'learning_rate': 0.1,           # Not used in RF, kept for API consistency
        }
        
        
        # Initialize different models based on config
        # This allows flexibility while keeping RandomForest as default
        
        model_type = self.config['model_type'].lower()
        
        if model_type == 'random_forest':
            # RandomForest with optimized hyperparameters
            # Random Forest = Ensemble of independent decision trees
            # Each tree trained on random sample of features/data
            self.model = RandomForestRegressor(
                n_estimators=self.config['n_estimators'],  # 100 trees
                max_depth=self.config['max_depth'],        # Limit tree depth
                random_state=random_state,                 # Reproducibility
                n_jobs=-1                                  # Use all CPU cores
            )
            
        elif model_type == 'ridge':
            # Ridge Regression (linear model with L2 penalty)
            self.model = Ridge(alpha=1.0, random_state=random_state)
            
        elif model_type == 'xgboost':
            # XGBoost (gradient boosting)
            self.model = xgb.XGBRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
            
        elif model_type == 'gradient_boosting':
            # Sklearn Gradient Boosting
            self.model = GradientBoostingRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                random_state=random_state
            )
            
        elif model_type == 'svr':
            # ALTERNATIVE: Support Vector Regression
            self.model = SVR(kernel='rbf', C=1.0)
            
        elif model_type == 'linear':
            # ALTERNATIVE: Simple Linear Regression (baseline)
            self.model = LinearRegression()
            
        else:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'\n"
                f"Choose from: 'random_forest', 'ridge', 'xgboost', 'gradient_boosting', 'svr', 'linear'"
            )
        
        # PIPELINE SETUP
        # Combine preprocessing + model into single pipeline
        # Ensures data flows: Raw Features ,Scaler, Model
        
        self.scaler = StandardScaler()  # Normalize features to mean=0, std=1
        self.pipeline = Pipeline([
            ('scaler', self.scaler),      # Step 1: Normalize
            ('model', self.model)          # Step 2: Train/predict
        ])
        
        # ====== STATE INITIALIZATION ======
        # Initialize tracking variables
        
        self.is_fitted = False            # Model not trained yet
        self.feature_names = None         # Will be populated during fit
        self.mean_y_fallback = 0.0        # Fallback value for predictions


    def _compute_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE CATEGORIES:
        1. MOMENTUM (5 features): How recent returns persist
        2. TREND (6 features): Price relative to moving averages
        3. VOLATILITY (3 features): Market uncertainty measures
        4. OSCILLATORS (3 features): Technical indicators (RSI, MACD)
        
        PARAMETERS:
        
        X : pd.DataFrame
            Input price data with columns: Close, High, Low (at minimum)
            - Close: Daily closing price
            - High: Daily high price (optional, defaults to Close)
            - Low: Daily low price (optional, defaults to Close)
        
        RETURNS:
            pd.DataFrame
            DataFrame with engineered features
            Index: Same as input (preserves alignment)
            Columns: ['log_ret_lag_1', 'log_ret_lag_2', ..., 'macd']
            Values: Normalized technical indicators (mostly in [-1, 1] range)
        
        WALK-FORWARD SAFETY:
        
         All features use only past/present data
         No future prices used (prevents look-ahead bias)
         Safe for backtesting and production use
        """
        
        # EXTRACTING PRICE COLUMNS 
        # Find columns case-insensitively (handles different naming conventions)
        
        close_col = next((c for c in X.columns if c.lower() == 'close'), None)
        high_col = next((c for c in X.columns if c.lower() == 'high'), None)
        low_col = next((c for c in X.columns if c.lower() == 'low'), None)
        
        # ValidatING required Close column
        if close_col is None:
            raise ValueError(
                "DataFrame must contain 'Close' column\n"
                f"Available columns: {list(X.columns)}"
            )
        
        # ExtractING price series (fallback to Close if High/Low missing)
        prices = X[close_col].copy()
        high = X[high_col].copy() if high_col else prices
        low = X[low_col].copy() if low_col else prices
        
        # CALCULATING LOG RETURNS 
        # Log returns = log(price_t / price_{t-1})
        # More stable than simple returns for modeling
        
        log_returns = np.log(prices / prices.shift(1)).fillna(0)
        
        # Initialize features dataframe
        features = pd.DataFrame(index=X.index)
        
        # FEATURES THAT ARE IN GROUP 1: MOMENTUM FEATURES (5 features)
        # How much past returns influence current direction
        # Add lagged returns (past 5 days)
        for lag in range(1, self.config['n_lags'] + 1):
            features[f'log_ret_lag_{lag}'] = log_returns.shift(lag).fillna(0)
        
        # FEATURES THAT ARE IN GROUP 2: TREND FEATURES (6 features)
        # Direction and magnitude of trend relative to moving averages
        
        # Calculating Simple Moving Averages (SMA)
        # SMA = average closing price over N days
        sma_short = prices.rolling(
            window=self.config['sma_short'], 
            min_periods=1
        ).mean()
        sma_medium = prices.rolling(
            window=self.config['sma_medium'], 
            min_periods=1
        ).mean()
        sma_long = prices.rolling(
            window=self.config['sma_long'], 
            min_periods=1
        ).mean()
        
        # Features Price relative to SMAs
        features['close_sma_short_ratio'] = (prices / (sma_short + 1e-8)).fillna(1.0)
        features['close_sma_medium_ratio'] = (prices / (sma_medium + 1e-8)).fillna(1.0)
        features['close_sma_long_ratio'] = (prices / (sma_long + 1e-8)).fillna(1.0)
        
        # Features Distance from SMA in %
        features['sma_short_dist'] = (
            (prices - sma_short) / (sma_short + 1e-8)
        ).fillna(0)
        features['sma_long_dist'] = (
            (prices - sma_long) / (sma_long + 1e-8)
        ).fillna(0)
        
        # Features SMA Crossover (bullish signal)
        features['sma_crossover'] = (sma_short > sma_long).astype(float).fillna(0.5)
        
        # FEATURES THAT ARE IN GROUP 3: VOLATILITY FEATURES (3 features)
        # Market uncertainty how much price fluctuates
        
        # Features Historical Volatility
        # Standard deviation of recent returns (risk measure)
        features['volatility'] = log_returns.rolling(
            window=10, 
            min_periods=1
        ).std().fillna(0)
        
        # Features ATR (Average True Range)
        # Captures intraday volatility (gaps and limit moves)
        # TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        tr1 = high - low
        tr2 = np.abs(high - prices.shift(1))
        tr3 = np.abs(low - prices.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(
            window=self.config['atr_length'], 
            min_periods=1
        ).mean()
        features['atr_normalized'] = (atr / (prices + 1e-8)).fillna(0)
        
        # Features Bollinger Bands Position
        bb_middle = prices.rolling(
            window=self.config['bb_length'], 
            min_periods=1
        ).mean()
        bb_std = prices.rolling(
            window=self.config['bb_length'], 
            min_periods=1
        ).std().fillna(0.01)
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        features['bb_position'] = (
            (prices - bb_lower) / (bb_upper - bb_lower + 1e-8)
        ).fillna(0.5)
        
        # FEATURE THAT ARE IN GROUP 4: OSCILLATORS (3 features)
        # Momentum indicators that are overbought/oversold conditions
        
        # Features RSI (Relative Strength Index)
        # Measures momentum: values 0-1 (0.5 = neutral)
        rsi = ta.rsi(prices, length=self.config['rsi_length'])
        features['rsi'] = (rsi / 100.0).fillna(0.5) if rsi is not None else 0.5
        
        # Features Momentum (5-day)
        # Simple momentum: how much price has changed in 5 days
        features['momentum'] = ((prices / prices.shift(5)) - 1).fillna(0)
        
        # Features MACD (Moving Average Convergence Divergence)
        macd_result = ta.macd(
            prices,
            fast=self.config['macd_fast'],
            slow=self.config['macd_slow'],
            signal=self.config['macd_signal']
        )
        if macd_result is not None and len(macd_result) > 0:
            # Get MACD line (first column of result)
            features['macd'] = macd_result.iloc[:, 0].fillna(0)
        else:
            # Fallback if MACD calculation fails
            features['macd'] = 0.0
        
        # CLEANING AND CHECKING VALIDATION
        # Ensure data quality before returning to model
        # Forward fill (propagate last valid observation forward)
        # Then fill remaining NaNs with 0
        features = features.ffill().fillna(0)
        
        # Replace infinite values with 0 (prevents model errors)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Reindex to match original X (handles missing indices gracefully)
        features = features.reindex(X.index).fillna(0)
        
        # Store feature names for reference and diagnostics
        self.feature_names = features.columns.tolist()
        
        return features


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta: dict | None = None):
        """
        TRAINING: Fitting RandomForest model on training data
        
        PARAMETERS:
        X_train : pd.DataFrame
            Training features (OHLCV price data)
            - Shape: (n_samples, n_features like Open, High, Low, Close, Volume)
            - Index: DatetimeIndex for time-series alignment
            - Important: Only contains past data (no future leakage)
        
        y_train : pd.Series
            Training targets (log returns for next 5 days)
            - Shape: (n_samples,)
            - Values: Log returns (typically -0.1 to +0.1)
            - Index: DatetimeIndex (must align with X_train after feature engineering)
        
        ALGORITHM:
        1. Sort data by date (ensure time order)
        2. Engineer 17 features from raw OHLCV
        3. Align features with targets (handle missing indices)
        4. Check data validity
        5. Fit RandomForest (learning from data)
        6. Store mean return as fallback
        7. Set is_fitted flag
        """
        # Chronological step
        # Important for time-series data to avoid look-ahead bias
        X_train = X_train.sort_index()
        
        # Engineer features from raw price data
        X_features = self._compute_features(X_train)
        
        # Align features and targets
        # After feature engineering, features may have different index
        # Find common indices between features and targets
        common_idx = X_features.index.intersection(y_train.index)
        X_features = X_features.loc[common_idx]
        y_train_aligned = y_train.loc[common_idx]
        
        # Handle edge case (no common indices)
        # This shouldn't happen in normal operation, but be defensive
        if len(X_features) == 0:
            self.mean_y_fallback = (
                float(y_train.mean()) if np.isfinite(y_train.mean()) else 0.0
            )
            self.is_fitted = False
            return self
        
        #Fit the pipeline (scaler + model)
        self.pipeline.fit(X_features, y_train_aligned)
        
        #Mark as fitted and store fallback mean
        self.is_fitted = True
        self.mean_y_fallback = float(y_train_aligned.mean())
        
        return self


    def predict(self, X: pd.DataFrame, meta: dict | None = None) -> pd.Series:
        
        # Sort test data chronologically
        X = X.sort_index()
        #Calling All Features
        X_features = self._compute_features(X)
        
        # Return fallback if model not fitted
        if not self.is_fitted or len(X_features) == 0:
            return pd.Series(
                self.mean_y_fallback,
                index=X.index,
                name='y_pred'
            )
        
        # Use pipeline to scale and predict
        y_pred_values = self.pipeline.predict(X_features)
        
        # STEP 5: Create Series with feature indices
        y_pred = pd.Series(
            y_pred_values,
            index=X_features.index,
            name='y_pred'  # ← REQUIRED by mltester API
        )
        
        # Reindex to original X (fill gaps with mean)
        # Handles case where feature engineering removes some dates
        y_pred = y_pred.reindex(X.index).fillna(self.mean_y_fallback)
        
        return y_pred
