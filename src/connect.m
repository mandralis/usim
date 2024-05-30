function myScope = connect()
    % connect with usb
    myScope = oscilloscope('USB::0x0699::0x0408::C012926::INSTR');

    % connect with ethernet
    % myScope = oscilloscope('TCPIP::169.254.2.50::INSTR');
end