function A = get_wavelet_amplitudes_from_signal(wavelets,signal,x_start)
    
    N = size(signal,1);
    A = zeros(length(x_start),Nsignals);
    M = collocation_matrix(wavelets,N,x_start);
    for i=1:Nsignals
        A(:,i) = pinv(M)*signal(:,i);
    end

end