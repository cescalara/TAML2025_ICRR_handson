data{
  int N;
  vector[N] x;
}

parameters { 
  real mu;
  real<lower=0> sigma;
}

model {
  mu ~ normal(10, 0.1);
  sigma ~ normal(1, 0.1);

  x ~ normal(mu, sigma);
}
