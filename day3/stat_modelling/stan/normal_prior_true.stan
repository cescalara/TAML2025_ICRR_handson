data{
  int N;
  vector[N] x;
}

parameters { 
  real mu;
  real<lower=0> sigma;
}

model {
  mu ~ normal(5, 0.1);
  sigma ~ normal(3, 0.1);

  x ~ normal(mu, sigma);
}
