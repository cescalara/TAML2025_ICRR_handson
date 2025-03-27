data{

  int N;
  vector[N] x_obs;
  vector[N] y_obs;
  real sigma;
}

parameters { 

  real m;
  real c;

}

transformed parameters {

  vector[N] y;

  y = m * x_obs + c;
  
}

model {
  
  y_obs ~ normal(y, sigma);

}
