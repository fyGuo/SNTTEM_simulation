boot_var <- function(df,ps1= working_ps1, ps0=working_ps0,mu01= working_mu01,mu12= working_mu12,mu02=
  working_mu02, iter) {
  n <- nrow(df)
  est_psi <- data.frame(matrix(NA, nrow = iter, ncol = 3))
  for (i in 1:iter) {
    sample_id <- sample(1:n, size = n, replace = TRUE) # bootstrap sample
    temp <- df[sample_id, ]
    # then we do the same estimation procedure as in the main sample

    id_1 <- sample(1:n, size = n/2, replace = FALSE)
    id_2 <- setdiff(1:n, id_1)

    temp <- working_model( temp, id_1, id_2, ps1, ps0 , mu01, mu12 , mu02) # apply the working models
    x <- as.matrix( temp)
    g <- function(theta, x){
      cbind(x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(x[,"Y1"]-gamma01(x[,"L0"], theta[1])*x[,"A0"]-x[,"mu01"]),
            x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))*(x[,"Y2"]-x[,"mu12"])+x[,"mu12"]-x[,"mu02"]-gamma02(x[,"L0"], theta[2])*x[,"A0"]),
            x[,"L0"]*(x[,"A0"]-x[,"ps0"])*(((1-x[,"A1"])/(1-x[,"ps1"]))^(1-x[,"I1"])*(x[,"Y2"]-gamma12(x[,"L1"], theta[3])*x[,"A1"] - x[,"mu12"]) + x[,"mu12"] - gamma02(x[,"L0"], theta[2])*x[,"A0"] - x[,"mu02"]),
            x[,"I1"]*x[,"L1"]*(x[,"A1"]-x[,"ps1"])*(x[,"Y2"]-gamma12(x[,"L1"], theta[3])*x[,"A1"] - x[,"mu12"])
      )
      } %>% return()
    mod <- gmm(g, x = x, t0 = c(1,1,1), type = "cue")
    est_psi[i,] <- mod$coefficients
  }
  vcov <- cov(est_psi)
}
