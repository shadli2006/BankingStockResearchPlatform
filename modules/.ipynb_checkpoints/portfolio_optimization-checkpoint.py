import numpy as np
import pandas as pd
from scipy.optimize import minimize
import random

def optimize_portfolio(results, method, target, health_scores=None):
    """Optimasi portofolio berdasarkan metode yang dipilih"""
    if results.empty:
        return {}, 0, 0
    
    # Buat pivot table untuk return
    returns_df = results.pivot_table(index='Date', columns='Bank', values=target).pct_change().dropna()
    
    if returns_df.empty or returns_df.shape[1] == 0:
        return {}, 0, 0
    
    # Siapkan skor kesehatan
    if health_scores is None:
        health_scores = pd.Series(1, index=returns_df.columns)
    else:
        # Pastikan semua bank memiliki skor
        missing_banks = returns_df.columns.difference(health_scores.index)
        if not missing_banks.empty:
            health_scores = pd.concat([health_scores, pd.Series(0, index=missing_banks)])
        health_scores = health_scores[returns_df.columns]  # Urutkan sesuai kolom
    
    if method == 'Genetic Algorithm':
        return genetic_optimization(returns_df, health_scores)
    elif method == 'Markowitz':
        return markowitz_optimization(returns_df)
    elif method == 'Markowitz + GA':
        # Kombinasi Markowitz + Genetic Algorithm
        weights_m, return_m, risk_m = markowitz_optimization(returns_df)
        weights_ga, return_ga, risk_ga = genetic_optimization(returns_df, health_scores)
        
        # Gabungkan hasil
        combined_weights = {}
        for bank in set(weights_m.keys()) | set(weights_ga.keys()):
            w_m = weights_m.get(bank, 0)
            w_ga = weights_ga.get(bank, 0)
            combined_weights[bank] = (w_m + w_ga) / 2
        
        # Normalisasi
        total = sum(combined_weights.values())
        for bank in combined_weights:
            combined_weights[bank] /= total
        
        # Hitung return dan risiko
        returns = returns_df.values
        portfolio_return = np.sum(returns_df.mean().values * np.array(list(combined_weights.values()))) * 252
        cov_matrix = np.cov(returns, rowvar=False)
        portfolio_risk = np.sqrt(np.dot(np.array(list(combined_weights.values())).T, 
                                  np.dot(cov_matrix, np.array(list(combined_weights.values()))))) * np.sqrt(252)
        
        return combined_weights, portfolio_return, portfolio_risk

def genetic_optimization(returns_df, health_scores):
    """Optimasi portofolio dengan Algoritma Genetika"""
    n_assets = returns_df.shape[1]
    returns = returns_df.values
    
    if n_assets == 1:
        bank_name = returns_df.columns[0]
        return {bank_name: 1.0}, returns.mean() * 252, 0.0
    
    # Parameter algoritma genetika
    population_size = 50
    num_generations = 100
    mutation_rate = 0.1
    
    # Inisialisasi populasi
    population = []
    for _ in range(population_size):
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        population.append(weights)
    
    # Fungsi fitness (Sharpe ratio + faktor kesehatan)
    def fitness(weights):
        portfolio_return = np.sum(returns.mean(axis=0) * weights) * 252
        cov_matrix = np.cov(returns.T)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        if portfolio_risk < 1e-8:
            return 0
        
        sharpe_ratio = portfolio_return / portfolio_risk
        
        # Faktor kesehatan
        health_factor = np.sum(health_scores.values * weights) / np.sum(weights)
        
        return sharpe_ratio * (1 + health_factor)
    
    # Evolusi populasi
    for _ in range(num_generations):
        # Evaluasi fitness
        fitness_scores = [fitness(ind) for ind in population]
        
        # Seleksi (turnamen)
        new_population = []
        for _ in range(population_size):
            # Pilih 3 individu acak
            tournament = random.sample(range(population_size), 3)
            # Pilih yang fitness-nya terbaik
            winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
            new_population.append(population[winner].copy())
        
        # Crossover (BLX-alpha)
        for i in range(0, population_size, 2):
            if i+1 < population_size:
                parent1 = new_population[i]
                parent2 = new_population[i+1]
                
                alpha = 0.5
                child1 = np.zeros(n_assets)
                child2 = np.zeros(n_assets)
                
                for j in range(n_assets):
                    d = abs(parent1[j] - parent2[j])
                    low = min(parent1[j], parent2[j]) - alpha * d
                    high = max(parent1[j], parent2[j]) + alpha * d
                    
                    child1[j] = np.random.uniform(low, high)
                    child2[j] = np.random.uniform(low, high)
                
                # Normalisasi
                child1 /= child1.sum()
                child2 /= child2.sum()
                
                new_population[i] = child1
                new_population[i+1] = child2
        
        # Mutasi
        for i in range(population_size):
            if random.random() < mutation_rate:
                # Mutasi Gaussian
                mutation = np.random.normal(0, 0.1, n_assets)
                new_population[i] = np.clip(new_population[i] + mutation, 0, 1)
                new_population[i] /= new_population[i].sum()
        
        population = new_population
    
    # Pilih individu terbaik
    fitness_scores = [fitness(ind) for ind in population]
    best_idx = np.argmax(fitness_scores)
    best_weights = population[best_idx]
    
    # Hitung return dan risiko
    portfolio_return = np.sum(returns.mean(axis=0) * best_weights) * 252
    cov_matrix = np.cov(returns.T)
    portfolio_risk = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights))) * np.sqrt(252)
    
    weights_dict = dict(zip(returns_df.columns, best_weights))
    return weights_dict, portfolio_return, portfolio_risk

def markowitz_optimization(returns_df):
    """Optimasi portofolio dengan Markowitz Mean-Variance"""
    n_assets = returns_df.shape[1]
    
    if n_assets == 0:
        return {}, 0, 0
        
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    initial_weights = np.ones(n_assets) / n_assets
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        weights = initial_weights
    else:
        weights = result.x
    
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_risk = objective(weights)
    
    weights_dict = dict(zip(returns_df.columns, weights))
    return weights_dict, portfolio_return, portfolio_risk