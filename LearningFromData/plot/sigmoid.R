sigmoid = function(x, w, b) {
  1 / (1 + exp(-w*x + b))
}

x  <- seq(-5, 5, 0.1)
y1 <- sigmoid(x,1, 0)
y2 <- sigmoid(x,2, 0)
y3 <- sigmoid(x,5, 0)
y4 <- sigmoid(x,100, 0)
y5 <- sigmoid(x,0.01, 0)
y6 <- sigmoid(x,-1, 0)

matplot(cbind(x,x,x,x,x,x), cbind(y1,y2,y3,y4,y5,y6), type='l', lty=cbind(1,1,1), xlab="x", ylab="y")
legend('bottomright', inset=.06, legend=c("sigmoid(1*x)", "sigmoid(2*x)", "sigmoid(5*x)", "sigmoid(100*x)", "sigmoid(0.01*x)", "sigmoid(-1*x)"), bty="n", cex=0.8, pt.cex = 0.8, lty=1, lwd=2, col=1:3)



y1 <- sigmoid(x,1, -2)
y2 <- sigmoid(x,1, -1)
y3 <- sigmoid(x,1, 0)
y4 <- sigmoid(x,1, 1)
y5 <- sigmoid(x,1, 2)
matplot(cbind(x,x,x,x,x), cbind(y1,y2,y3,y4,y5), type='l', lty=cbind(1,1,1), xlab="x", ylab="y")
legend('bottomright', inset=.06, legend=c("sigmoid(1*x-2)", "sigmoid(1*x-1)", "sigmoid(1*x)", "sigmoid(1*x+1)", "sigmoid(1*x+2)"), bty="n", cex=0.8, pt.cex = 0.8, lty=1, lwd=2, col=1:3)
