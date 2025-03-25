// Simple Beta-Binomial Stan model (no weights)

// Bayesian integration model relying on a beta-binomial distribution
// to preserve all uncertainty
// All evidence is taken at face value (equal weights)
data {
  int<lower=1> N;                      // Number of decisions
  array[N] int<lower=0, upper=7> choice; // Choices (0=untrustworthy, 7=santa klaús)
  array[N] int<lower=0> alpha_likelihood;         // Direct evidence (blue marbles)
  array[N] int<lower=0> total;        // Total direct evidence (total marbles)
  array[N] int<lower=0> alpha_likelihood_social;         // Social evidence (blue signals)
  array[N] int<lower=0> total_social;        // Total social evidence (total signals)
}

parameters {
  real<lower=0> alpha_prior;         // Alpha prior
  real<lower=0> beta_prior;          // Beta prior
}


model {
  // Each observation is a separate decision
  target += lognormal_lpdf(alpha_prior|0, 1);
  target += lognormal_lpdf(beta_prior|0, 1);
  
  
  for (i in 1:N) {
    // Calculate Beta parameters for posterior belief distribution
    real alpha_post = alpha_prior + alpha_likelihood[i] + alpha_likelihood_social[i];
    real beta_post = beta_prior + (total[i] - alpha_likelihood[i]) + (total_social[i] - alpha_likelihood_social[i]);
    
    // Use beta_binomial distribution which integrates over all possible values
    // of the rate parameter weighted by their posterior probability
    target += beta_binomial_lpmf(choice[i] | 7, alpha_post, beta_post);
  }
}


generated quantities {
  // Log likelihood for model comparison
  vector[N] log_lik;
  
  // Prior and posterior predictive checks
  array[N] int prior_pred_choice;
  array[N] int posterior_pred_choice;
  
  for (i in 1:N) {
    // For prior predictions, use uniform prior (Beta(1,1))
    prior_pred_choice[i] = beta_binomial_rng(7, alpha_prior, beta_prior);
    
    // For posterior predictions, use integrated evidence
    real alpha_post = alpha_prior + alpha_likelihood[i] + alpha_likelihood_social[i];
    real beta_post = beta_prior + (total[i] - alpha_likelihood[i]) + (total_social[i] - alpha_likelihood_social[i]);
    
    // Generate predictions using the complete beta-binomial model
    posterior_pred_choice[i] = beta_binomial_rng(7, alpha_post, beta_post);
    
    // Log likelihood calculation using beta-binomial
    log_lik[i] = beta_binomial_lpmf(choice[i] | 7, alpha_post, beta_post);
  }
}




