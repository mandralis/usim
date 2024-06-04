function out = get_collocation_matrix(wavelets,N,x_start)
    out = zeros(N,length(x_start)); % collocation matrix
    for k = 1:length(wavelets)
        out(:,k) = get_padded_wavelet(wavelets{k},N,x_start(k));
    end
end