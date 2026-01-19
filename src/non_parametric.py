#!/usr/bin/env python3
"""
Non-Parametric Gaussian Copula Synthesizer

This module implements a custom Gaussian Copula Synthesizer that uses 
Empirical CDFs (ECDF) with specific handling for continuous, integer, 
and categorical variables to model marginal distributions.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

__all__ = ["NonParamGaussianCopulaSynthesizer"]

LOGGER = logging.getLogger(__name__)

class NonParamGaussianCopulaSynthesizer:
    """
    Custom Gaussian Copula Synthesizer using Empirical CDF for marginals.
    Structure inspired by SDV's GaussianCopulaSynthesizer but fully self-contained.
    """
    
    def __init__(self, enforce_min_max_values=True, epsilon=1.0):
        self.enforce_min_max_values = enforce_min_max_values
        self.epsilon = epsilon
        self._model_state = {}
        self._fitted = False
        
    def fit(self, data, epsilon=None):
        """Fit the model to the table."""
        self._validate_data(data)
        eps = epsilon if epsilon is not None else self.epsilon
        LOGGER.info(f"Fitting with differential privacy epsilon={eps}...")
        LOGGER.info("Fitting NonParamGaussianCopulaSynthesizer...")
        
        # 1. Learn Marginals & Transform to Z (The "Fit" step)
        self._model_state = self._learn_distributions_and_correlation(data, epsilon=eps)
        self._fitted = True
        LOGGER.info("Model fitted successfully.")

    def sample(self, num_rows, seed=None):
        """Sample the indicated number of rows from the model."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call `fit` first.")
            
        LOGGER.info(f"Sampling {num_rows} rows...")
        return self._generate_samples(num_rows, seed)

    # ============================================================
    # FIXED: Save as an Object (Compatible with SDV loaders)
    # ============================================================

    def save(self, filepath):
        """
        Save the ENTIRE object instance to a pickle file.
        This allows sdv.load_synthesizer to work correctly.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Cannot save.")
            
        with open(filepath, 'wb') as f:
            # FIX: Dump 'self' (the object), NOT 'self.__dict__' (the dict)
            pickle.dump(self, f)
            
        LOGGER.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load the object state from a pickle file into this instance.
        """
        with open(filepath, 'rb') as f:
            # This reads the Pickled Object
            loaded_instance = pickle.load(f)
        
        # If you used sdv.load_synthesizer externally, you wouldn't need this.
        # But to support model.load(), we transfer the state:
        if isinstance(loaded_instance, dict):
            # Fallback for old 'bad' files (dictionaries)
            self.__dict__.update(loaded_instance)
        else:
            # Standard behavior: loaded_instance is an object
            self.__dict__.update(loaded_instance.__dict__)
            
        self._fitted = True
        LOGGER.info(f"Model loaded from {filepath}")

    # ============================================================
    # Internal Logic
    # ============================================================

    def _validate_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Data is empty.")

    def _learn_distributions_and_correlation(self, df, epsilon=None):
        """
        Learns the marginal distributions and correlation matrix with DP budget splitting.
        """
        if epsilon is None or epsilon <= 0:
            eps_marginal = None
            eps_corr = None
        else:
            eps_marginal = epsilon * 0.5
            eps_corr = epsilon * 0.5
        
        rng = np.random.default_rng()
        z_df = pd.DataFrame(index=df.index, columns=df.columns)
        marginals = {}

        for col in df.columns:
            series = df[col]
            valid_data = series.dropna()
            nan_frac = series.isna().mean()
            dtype = series.dtype
            if pd.api.types.is_numeric_dtype(dtype):
                is_integer = np.allclose(valid_data % 1, 0) if len(valid_data) > 0 else False
                marginals[col] = {
                    "type": "integer" if is_integer else "continuous",
                    "sorted_values": np.sort(valid_data.values),
                    "nan_frac": nan_frac,
                    "dtype": dtype,
                    "dtype_name": str(dtype)
                }
                # Use the split budget
                u = self._empirical_cdf_continuous(series, rng, epsilon=eps_marginal)

            else:
                sorted_labels = np.sort(valid_data.unique()).tolist()
                val_counts = valid_data.value_counts()
                counts_in_order = [val_counts.get(l, 0) for l in sorted_labels]
                marginals[col] = {
                    "type": "categorical",
                    "labels": sorted_labels,
                    "counts": counts_in_order,
                    "nan_frac": nan_frac,
                    "dtype": dtype,
                    "dtype_name": str(dtype)
                }
                # Use the split budget
                u = self._empirical_cdf_categorical(series, sorted_labels, rng, epsilon=eps_marginal)

            z_df[col] = self._uniform_to_gaussian(u)
            
        correlation_matrix = z_df.corr(method='pearson').fillna(0.0)
        
        if eps_corr:
            n = len(z_df)
            noise_scale = 2.0 / (n * eps_corr)
            noise = rng.laplace(0, noise_scale, size=correlation_matrix.shape)
            noisy_corr = correlation_matrix + noise
            noisy_corr = (noisy_corr + noisy_corr.T) / 2
            np.fill_diagonal(noisy_corr.values, 1.0)
            fixed_corr_values = self._get_nearest_correlation_matrix(noisy_corr.values)
            final_corr_df = pd.DataFrame(fixed_corr_values, columns=df.columns, index=df.columns)
        else:
            final_corr_df = correlation_matrix  
        
        return {
            "correlation_matrix": final_corr_df,
            "marginals": marginals,
            "columns": df.columns.tolist()
        }
    
    def _get_nearest_correlation_matrix(self, matrix):
        """
        Finds the nearest Positive Semi-Definite (PSD) matrix.
        Ensures the matrix can be used for Cholesky decomposition.
        """
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.clip(vals, 1e-8, None)
        new_mat = (vecs * vals) @ vecs.T
        d = np.sqrt(np.diag(new_mat))
        new_mat = new_mat / np.outer(d, d)
        
        return new_mat

    def _generate_samples(self, n_samples, seed=None):
        rng = np.random.default_rng(seed)
        
        state = self._model_state
        columns = state['columns']
        marginals = state['marginals']
        corr_matrix = state['correlation_matrix']

        # 1. Generate Independent Z
        z_ind = rng.standard_normal(size=(n_samples, len(columns)))
        z_ind_df = pd.DataFrame(z_ind, columns=columns)

        # 2. Apply Correlation
        z_correlated = self._apply_correlation(z_ind_df, corr_matrix)

        # 3. Inverse Transform
        synthetic_data = pd.DataFrame(index=range(n_samples), columns=columns)
        
        for col in columns:
            meta = marginals[col]
            u_samples = self._gaussian_to_uniform(z_correlated[col])
            
            if meta['type'] == 'continuous':
                synthetic_data[col] = self._inverse_ecdf_continuous(u_samples, meta)
            elif meta['type'] == 'integer':
                synthetic_data[col] = self._inverse_ecdf_integer(u_samples, meta)
            elif meta['type'] == 'categorical':
                synthetic_data[col] = self._inverse_ecdf_categorical(u_samples, meta)

            # Restore Dtypes
            try:
                dtype_name = meta.get('dtype_name', '')
                if 'int' in dtype_name.lower() or meta['type'] == 'integer':
                    synthetic_data[col] = synthetic_data[col].round().astype(meta['dtype'])
                else:
                    synthetic_data[col] = synthetic_data[col].astype(meta['dtype'])
            except Exception as e:
                LOGGER.warning(f"Could not cast column '{col}': {e}")

        return synthetic_data

    # ============================================================
    # Math Helper Methods
    # ============================================================
    
    def _apply_correlation(self, z_ind, corr_matrix):
        try:
            L = np.linalg.cholesky(corr_matrix.values)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(corr_matrix.values)
            vals = np.clip(vals, 1e-8, None)
            R_pd = (vecs * vals) @ vecs.T
            D = np.sqrt(np.diag(R_pd))
            R_pd = R_pd / np.outer(D, D)
            L = np.linalg.cholesky(R_pd)
            
        X_arr = z_ind.values @ L.T
        return pd.DataFrame(X_arr, columns=z_ind.columns)

    def _gaussian_to_uniform(self, z):
        return norm.cdf(np.asarray(z, float))

    def _uniform_to_gaussian(self, u):
        return norm.ppf(np.asarray(u, float))
    
    def _empirical_cdf_continuous(self, column, rng, epsilon=1.0, integer_tolerance=1e-12):
        arr = np.asarray(column, float)
        mask = ~np.isnan(arr)
        valid = arr[mask]
        if valid.size == 0: return np.full(arr.shape, np.nan)

        # Determine if we treat as discrete integers or continuous float
        is_int = np.allclose(valid, np.round(valid), atol=integer_tolerance)
        u = np.full(arr.shape, np.nan)

        if is_int:
            uniques, counts = np.unique(valid, return_counts=True)
            counts = counts.astype(float)
            if epsilon:
                noise = rng.laplace(0, 1.0 / epsilon, size=len(counts))
                counts = np.maximum(counts + noise, 1e-5)
                
            
            p = counts / counts.sum()
            P = np.cumsum(p)
            L = np.insert(P[:-1], 0, 0)
            
            idx_map = {val: i for i, val in enumerate(uniques)}
            indices = np.array([idx_map[v] for v in valid])
            u[mask] = L[indices] + rng.random(valid.size) * p[indices]
        else:
            if epsilon:
                # DP Path: Noisy Histograms provide a bounded-sensitivity ECDF
                counts, edges = np.histogram(valid, bins=100)
                noisy_counts = np.maximum(counts + rng.laplace(0, 1/epsilon, size=100), 0)
                cdf = np.insert(np.cumsum(noisy_counts) / (noisy_counts.sum() or 1), 0, 0)
                u[mask] = np.interp(valid, edges, cdf)
                if epsilon and epsilon < 0.01:
                    print(f"DEBUG: Raw counts sum: {counts.sum()}")
                    print(f"DEBUG: Noisy counts sum: {noisy_counts.sum()}")#################################
            else:
                # Standard Path: Fallback to exact ranks
                u[mask] = (rankdata(valid, method='average') - 0.5) / valid.size

        return np.clip(u, 1e-12, 1 - 1e-12)
        
    def _empirical_cdf_categorical(self, column, sorted_labels, rng, epsilon=1.0):
        # Ensure epsilon has a default value (1.0) so it's always private unless specified
        arr = np.asarray(column, dtype=object)
        arr_filled = np.array(["<NaN>" if pd.isna(v) else v for v in arr], dtype=object)
        
        # Logic to handle labels + NaN
        labels = sorted_labels + (["<NaN>"] if "<NaN>" not in sorted_labels else [])
        
        if arr_filled.size == 0: 
            return np.full(arr.shape, np.nan, dtype=float)

        # 1. Get raw counts
        uniq, cnt = np.unique(arr_filled, return_counts=True)
        count_map = dict(zip(uniq, cnt))
        counts = np.array([count_map.get(l, 0) for l in labels], dtype=float)

        # 2. Apply DP Noise
        if epsilon is not None:
            # Scale noise by 1/epsilon. Sensitivity is 1.
            noise = rng.laplace(0, 1.0 / epsilon, size=len(counts))
            counts = np.maximum(counts + noise, 1e-5) # Prevent zeros/negatives

        # 3. Probabilities and Cumulative Probabilities
        p = counts / counts.sum()
        P = np.cumsum(p)
        L = np.insert(P[:-1], 0, 0)
        
        # 4. Map back to vectors
        p_map = dict(zip(labels, p))
        L_map = dict(zip(labels, L))
        
        p_vec = np.array([p_map.get(v, 0.0) for v in arr_filled])
        L_vec = np.array([L_map.get(v, 0.0) for v in arr_filled])
        
        # Return the jittered U values
        return L_vec + rng.random(size=arr_filled.shape[0]) * p_vec

    def _inverse_ecdf_integer(self, u_values, meta):
        sorted_vals = meta['sorted_values']
        nan_frac = meta['nan_frac']
        u = np.asarray(u_values, float)
        n = len(sorted_vals)
        result = np.full(u.shape, np.nan, dtype=float)
        mask_nan = np.isnan(u)
        result[mask_nan] = np.nan

        if nan_frac > 0:
            mask_cat = (~mask_nan) & (u > 1.0 - nan_frac)
            result[mask_cat] = np.nan
        else:
            mask_cat = np.zeros_like(u, bool)

        mask_valid = ~(mask_nan | mask_cat)
        u_valid = u[mask_valid]
        denom = max(1.0 - nan_frac, 1e-12)
        u_adj = np.clip(u_valid / denom, 0, 1 - 1e-8)
        knots_u = (np.arange(1, n+1) - 0.5) / n
        x_cont = np.interp(u_adj, knots_u, sorted_vals)
        
        uniques = np.unique(sorted_vals)
        diffs = np.abs(x_cont[:, None] - uniques[None, :])
        result[mask_valid] = uniques[diffs.argmin(axis=1)]
        return result

    def _inverse_ecdf_categorical(self, u_values, meta):
        sorted_labels = list(meta['labels'])
        counts = list(meta['counts'])
        nan_frac = meta['nan_frac']
        if nan_frac > 0:
            n_nan = sum(counts) * nan_frac / (1.0 - nan_frac) if sum(counts)>0 else 0
            sorted_labels.append("<NaN>")
            counts.append(n_nan)

        u = np.asarray(u_values, float)
        out = np.full(u.shape, np.nan, dtype=object)
        mask_valid = ~np.isnan(u)
        if not np.any(mask_valid): return out

        u_adj = np.clip(u[mask_valid], 0.0, 1.0 - 1e-12)
        counts_arr = np.asarray(counts, float)
        if counts_arr.sum() <= 0: return out
        P = np.cumsum(counts_arr / counts_arr.sum())
        inds = np.clip(np.searchsorted(P, u_adj, side="right"), 0, len(sorted_labels) - 1)
        chosen = np.asarray(sorted_labels, dtype=object)[inds]
        chosen[chosen == "<NaN>"] = np.nan
        out[mask_valid] = chosen
        return out

    def _inverse_ecdf_continuous(self, u_values, meta):
        sorted_vals = meta['sorted_values']
        nan_frac = meta['nan_frac']
        u = np.asarray(u_values, float)
        n = len(sorted_vals)
        result = np.full(u.shape, np.nan, dtype=float)
        mask_nan = np.isnan(u)
        result[mask_nan] = np.nan

        if nan_frac > 0:
            mask_cat = (~mask_nan) & (u > 1.0 - nan_frac)
            result[mask_cat] = np.nan
        else:
            mask_cat = np.zeros_like(u, bool)

        mask_valid = ~(mask_nan | mask_cat)
        denom = max(1.0 - nan_frac, 1e-12)
        u_adj = np.clip(u[mask_valid] / denom, 1e-8, 1 - 1e-8)
        knots_u = (np.arange(1, n+1) - 0.5) / n
        result[mask_valid] = np.interp(u_adj, knots_u, sorted_vals)
        return result