import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from sklearn.base import BaseEstimator, CloneMixin

class ConformalPredictionWrapper(BaseEstimator, CloneMixin):
    def __init__(self, base_estimator, method="score", cv=5, random_state=42):
        """
        Wrapper for Conformal Prediction using MAPIE.
        
        Args:
            base_estimator: The underlying classifier (e.g., XGBoost).
            method: 'score' (default) or 'cumulated_score'.
            cv: Cross-validation folds for calibration (int) or 'prefit'.
            random_state: Random seed.
        """
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.mapie = None
        
    def fit(self, X, y):
        """
        Fits the MapieClassifier.
        If cv > 1, it performs cross-validation to calibrate.
        """
        # Ensure y is proper format
        if isinstance(y, pd.Series):
            y = y.values
            
        self.mapie = MapieClassifier(
            estimator=self.base_estimator,
            method=self.method,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.mapie.fit(X, y)
        return self
        
    def predict(self, X, alpha=0.1):
        """
        Predicts classes and prediction sets.
        
        Args:
            X: Features.
            alpha: Significance level (e.g., 0.1 for 90% confidence).
            
        Returns:
            y_pred: Standard point predictions.
            y_pis: Prediction sets (boolean mask: [n_samples, n_classes]).
                 True if class is in the set.
        """
        y_pred, y_pis = self.mapie.predict(X, alpha=alpha)
        
        # y_pis shape is (n_samples, n_classes, n_alpha)
        # We assume single alpha, so squeeze last dim if needed
        if y_pis.ndim == 3:
            y_pis = y_pis[:, :, 0]
            
        return y_pred, y_pis
        
    def get_certainty_mask(self, y_pis):
        """
        Returns a boolean mask where the model is 'Certain' (Set Size == 1).
        If Set Size > 1 (Uncertain) or Set Size == 0 (Empty), returns False.
        """
        set_sizes = np.sum(y_pis, axis=1)
        return set_sizes == 1

    def filter_signals(self, signals: pd.Series, X, alpha=0.1, class_map=None):
        """
        Filters an existing signal series using Conformal Prediction.
        Only keeps signals where the model predicts that SPECIFIC class as the ONLY possibility.
        
        Args:
            signals: Series of -1, 0, 1
            X: Features corresponding to signals
            alpha: Significance level
            class_map: Dict mapping signal values (-1, 0, 1) to model class indices (0, 1, 2)
        
        Returns:
            Filtered signals (pd.Series)
        """
        if self.mapie is None:
            raise ValueError("Model not fitted yet.")
            
        # Get Prediction Sets
        _, y_pis = self.predict(X, alpha=alpha)
        
        # Create mask
        # We need to map signal values to column indices in y_pis
        # Usually: Sell(-1) -> 0, Hold(0) -> 1, Buy(1) -> 2 (if sorted)
        # But it depends on the fitted classes
        
        if class_map is None:
            # Infer from mapie classes
            classes = self.mapie.classes_
            class_map = {c: i for i, c in enumerate(classes)}
            
        filtered_signals = signals.copy()
        
        for idx, signal_val in signals.items():
            if signal_val == 0:
                continue # We usually don't filter Holds, or maybe we do? 
                         # Let's focus on Trades (1 / -1)
            
            # Find integer index for this signal
            if signal_val not in class_map:
                # Unknown class? Filter it out to be safe
                filtered_signals.at[idx] = 0
                continue
                
            c_idx = class_map[signal_val]
            row_idx = X.index.get_loc(idx)
            
            # Check if this class is in the prediction set
            is_in_set = y_pis[row_idx, c_idx]
            
            # Check set size
            set_size = np.sum(y_pis[row_idx])
            
            # STRICT FILTER: 
            # 1. The signal class MUST be in the set.
            # 2. The set size MUST be 1 (No ambiguity).
            
            if not (is_in_set and set_size == 1):
                filtered_signals.at[idx] = 0
                
        return filtered_signals
