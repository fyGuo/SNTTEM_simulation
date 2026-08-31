library(SuperLearner)
library(xgboost)
library(ranger)

rf <- create.Learner("SL.ranger", params = list(num.trees = 500))
xgboot <- create.Learner("SL.xgboost", params = list(ntrees = 500, max_depth = 4,
                                                     shrinkage=0.1))
gam <- create.Learner("SL.gam", tune = list(df = 3:8))

source("Generate_data.R")
source("working_models_ml.R")

p_I1 <- 1
ps1_int <- 0
psi01 <- 1
psi02 <- 1
psi12 <- 1
p1 <- 0.05
df <- generate_data(20000, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, p_I1=p_I1, p1 = p1)

df <- df %>% mutate(
  ps1 = A0*p1 + (1-A0)*(1-p1),
  mu12 = exp(-2+L0+psi02*L0*A0),
  ps0 =plogis(-1+L0),
  mu02 = exp(-2+L0),
  varphi1 = (A1-ps1)*(Y2*exp(-gamma12(L1, psi12)*(A1-A0)) - mu12),
  varphi0 = (A0-ps0)*(((A0==A1)/((1-ps1)*(1-A0) + ps1*A0))^(1-I1)*exp(-gamma02(L0, psi02)*A0)*(Y2*exp(-gamma12(L1, psi12)*(A1-A0))-mu12)
                      +mu12*exp(-gamma02(L0, psi02)*A0)-mu02),
  varphi1_varphi0 = varphi1*varphi0,
  varphi1_2 = varphi1^2
)


mod1 <- SuperLearner(df$varphi1_varphi0, data.frame(L1 = df$L1,
                                                    A0 = df$A0,
                                                    L0 = df$L0),
                     family = gaussian(),
                     SL.library = c(rf$names, xgboot$names, gam$names))

mod2 <- SuperLearner(df$varphi1_2, data.frame(L1 = df$L1,
                                              A0 = df$A0,
                                              L0 = df$L0),
                     family = gaussian(),
                     SL.library = c(rf$names, xgboot$names, gam$names))

saveRDS(mod1, "mod1_ortho_p1_005.rds")
saveRDS(mod2, "mod2_ortho_p1_005.rds")

#
# fun_ortho <- function(mod1, mod2, newdata){
#   pred_cov <- predict(mod1,newdata)$pred
#
#   pred_var_inv <-  (predict(mod2,newdata)$pred)^(-1)
#
#   return(pred_cov*pred_var_inv)
# }
#
#
#
#
# ###################
# ####################
# # train efficient indexing functions
#
# df <- generate_data(20000, psi02 = psi02, psi12 = psi12, ps1_int = ps1_int, p_I1=p_I1)
#
# df <- df %>% mutate(
#   ps1 = plogis(ps1_int+0.5*A0+0.75*L1),
#   mu12 = exp(-2+L0+psi02*L0*A0),
#   ps0 =plogis(-1+L0),
#   mu02 = exp(-2+L0),
#   varphi1 = (A1-ps1)*(Y2*exp(-gamma12(L1, psi12)*(A1-A0)) - mu12),
#   varphi0 = (A0-ps0)*(((A0==A1)/((1-ps1)*(1-A0) + ps1*A0))^(1-I1)*exp(-gamma02(L0, psi02)*A0)*(Y2*exp(-gamma12(L1, psi12)*(A1-A0))-mu12)
#                       +mu12*exp(-gamma02(L0, psi02)*A0)-mu02),
#   varphi1_varphi0 = varphi1*varphi0,
#   varphi1_2 = varphi1^2,
#   varphi0_check = varphi0 - fun_ortho(mod1, mod2, data.frame(L1 = L1,
#                                                              A0 = A0,
#                                                              L0 = L0))*varphi1,
#   varphi0_check_2 = varphi0_check^2,
#   varphi1_der = -(A1-ps1)*exp(Y2*exp(-gamma12(L1, psi12)*(A1-A0)) - mu12)*(A1-A0)*(1+L1),
#   varphi0_der = -(A0-ps0)*(((A0==A1)/((1-ps1)*(1-A0) + ps1*A0))^(1-I1)*exp(-gamma02(L0, psi02)*A0)*(Y2*exp(-gamma12(L1, psi12)*(A1-A0))-mu12)*L0*A0
#                            +mu12*exp(-gamma02(L0, psi02)*A0)*L0*A0)
# )
#
# mod3 <- SuperLearner(df$varphi0_check_2, data.frame(L0 = df$L0),
#                      family = gaussian(),
#                      SL.library = c(rf$names, xgboot$names, gam$names))
# mod4 <- SuperLearner(df$varphi1_der, data.frame(L1 = df$L1,
#                                                 A0 = df$A0,
#                                                 L0 = df$L0),
#                      family = gaussian(),
#                      SL.library = c(rf$names, xgboot$names, gam$names))
#
# mod5 <- SuperLearner(df$varphi0_der, data.frame(L0 = df$L0),
#                      family = gaussian(),
#                      SL.library = c(rf$names, xgboot$names, gam$names))
#
#
# saveRDS(mod3, "mod3.rds")
# saveRDS(mod4, "mod4.rds")
# saveRDS(mod5, "mod5.rds")
#
#
# fun_d0_eff <- function(mod3, mod5, newdata){
#   return(predict(mod3,newdata)$pred^(-1)*(predict(mod5,newdata)$pred))
# }
#
# fun_d1_eff <- function(mod2, mod4, newdata){
#   return(predict(mod2,newdata)^(-1)$pred*(predict(mod4,newdata)$pred))
# }
