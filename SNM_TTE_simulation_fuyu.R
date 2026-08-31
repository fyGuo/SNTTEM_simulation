library(gesttools)
library(tidyverse)
N<-1000
res<-matrix(NA,N,4)

for(i in 1:N){
#Generate data - two time points
n<-500
U<-rnorm(n,0,1)
L0<-rnorm(n,1)
A0<-rbinom(n,1,plogis(1+L0))
L1<-rnorm(n,1+L0+0.5*A0+U)
A1<-rbinom(n,1,plogis(1+0.5*A0+0.75*L1))
Y<-rnorm(n,1+0.5*A0+A1+L1+L0+U,A1+0.25*abs(L1))

#Estimator 1 - conventional AIPW approach
ps0<-predict(glm(A0~L0,family = "binomial"),type = "response")
ps1<-predict(glm(A1~L1+A0,family = "binomial"),type = "response")
w1<-(1-A1)/(1-ps1)
i1<-rep(NA,n);i1[A0==0]=1;i1[A0==1]=0
mu1<-cbind(1,A0,L1,L0)%*%coef(lm(Y~A0+L1+L0,subset=A1==0))
mu0<-cbind(1,L0)%*%coef(lm(mu1~L0,subset=A0==0))
theta_1<-sum(i1*(A1-ps1)*(Y-mu1))/sum(i1*(A1-ps1)*A1)
theta_0<-sum((A0-ps0)*(w1*(Y-mu1)+mu1-mu0))/sum((A0-ps0)*A0)
res[i,1]<-theta_0

#Estimator 2 - g-estimation
res[i,2]<-sum((A0-ps0)*((w1+i1*(1/(1-ps1))*(A1-ps1))*(Y-theta_1*A1-mu1)+mu1-mu0))/sum((A0-ps0)*A0)

#Estimator 3 - Improved g-estimation via LS projection
delta_a<-(A0-ps0)*w1*(Y-mu1)
delta_b<--(A0-ps0)*i1*(A1-ps1)*(Y-theta_1*A1-mu1)*(1/(1-ps1))
delta_c<-coef(lm(delta_a~-1+delta_b))
res[i,3]<-sum((A0-ps0)*((w1+delta_c*i1*(1/(1-ps1))*(A1-ps1))*(Y-theta_1*A1-mu1)+mu1-mu0))/sum((A0-ps0)*A0)

#Estimator 4 - estimating the optimal projection
tmp_out_a<-(Y-mu1)^2
gamma_a<-cbind(1,A0,L1,L0)%*%coef(lm(tmp_out_a~A0+L1+L0,subset=(A1==0)))
tmp_out_b<-((A1-ps1)^2)*((Y-theta_1*A1-mu1)^2)
gamma_b<-predict(gam(tmp_out_b~A0+L1+L0))
gamma_c<-as.vector(as.vector(gamma_a)/gamma_b)
delta_b<--(A0-ps0)*i1*(A1-ps1)*(Y-theta_1*A1-mu1)*ps1*gamma_c
delta_c<-coef(lm(delta_a~-1+delta_b))
res[i,4]<-sum((A0-ps0)*((w1+delta_c*i1*gamma_c*(A1-ps1))*(Y-theta_1*A1-mu1)+mu1-mu0))/sum((A0-ps0)*A0)
}

#Truth=1
mean(res[,1]);mean(res[,2]);mean(res[,3]);mean(res[,4])
var(res[,1]);var(res[,2]);var(res[,3]);var(res[,4])

