function y = cmplx_pdf(x,mu,sigma)

X = real(mu);
Y = imag(mu);

y = 1/(2*pi*sigma)*exp(-((real(x) - X)^2 + (imag(x) - Y)^2)/(2*sigma));
end