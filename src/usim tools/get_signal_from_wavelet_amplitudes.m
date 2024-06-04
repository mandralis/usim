function predicted_signal = get_signal_from_wavelet_amplitudes()
    signal_idx = 1:length(signal);
    for i = 1:Nsignals
        signal_ = signal(signal_idx,i);
        prediction_ = M(signal_idx,:)*A(:,i);
    end
end