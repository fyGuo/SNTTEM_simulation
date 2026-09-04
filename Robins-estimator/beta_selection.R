ee_selection <- function(data) {
  function(theta){
    with(data,
         c((1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0) +
             (1+L0)*beta*(ps1)*(A0-ps0)*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1),
           (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
  }
}


beta_selection <- function(beta, data) {
  last_var <- 10000000000
  for (i in 1:length(beta)) {
    indexing_beta <- beta[i]
    ee_selection <- function(data) {
      function(theta){
        with(data,
             c((1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, theta[1])*A0) +
                 (1+L0)*indexing_beta*(ps1)*(A0-ps0)*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1),
               (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, theta[2])*A1 - mu1)))
      }
    }

    est_psi <-  m_estimate(ee_selection , data = data, root_control = setup_root_control(start = c(0,0)))@"estimates"
    # then we estimate the variance of the hat psi0
    data <- data %>% mutate(phi0_old = (1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, est_psi[1])*A0),
                  phi0 = (1+L0)*(A0-ps0)*((1-A1)/(1-ps1)*(Y-mu1)+mu1-mu0-gamma0(L0, est_psi[1])*A0) +
                    (1+L0)*indexing_beta*(ps1)*(A0-ps0)*(A1-ps1)*(Y-gamma1(L1, est_psi[1])*A1 - mu1),
                  phi1 = (1+L1)/(1+2*(varY_control + varY_control^2)*(1-ps1))*(A1-ps1)*(Y-gamma1(L1, est_psi[2])*A1 - mu1),
                  rho = cov(phi0,phi1),
                  L = mean((1+L0)*indexing_beta*(ps1)*(A0-ps0)*(1+L1)*(A1-ps1)*A1)/mean((1+L1)^2*(A1-ps1)*A1),
                  var_bench = mean(phi0_old^2),
                  var_new = mean(phi0^2)-2*L*rho + L^2*mean(phi1^2))
    now_var <- data$var_new[1]
    if (now_var <= last_var) {
      last_var <- now_var
      opt_beta <- beta[i]}
  }
  return(opt_beta)
}


