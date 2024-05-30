function trigger_arduino(a)
    writeDigitalPin(a,'D4',1);
    pause(0.1);
    writeDigitalPin(a,'D4',0);
end