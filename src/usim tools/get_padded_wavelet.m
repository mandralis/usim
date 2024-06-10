function [out] = get_padded_wavelet(wavelet,L,T)
% take in wavelet, length, and time and output a zero padded vector

% find size of wavlet
len = length(wavelet);

% make zero padding
out = zeros(L,1);

% add signal
out(T:T+len-1) = wavelet; 

end