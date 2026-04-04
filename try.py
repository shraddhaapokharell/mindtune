import time
import csv
import os

try:
    import serial
    Serial = serial.Serial
    SerialException = serial.SerialException
except (ImportError, AttributeError):
    print("PySerial is missing or a conflicting 'serial' package is installed.")
    print("Fix with:")
    print("  python -m pip uninstall -y serial")
    print("  python -m pip install pyserial")
    raise SystemExit(1)

# Setup
COM_PORT = "COM3"
BAUD_RATE = 57600
OUTPUT_FILE = "mindtune_full_eeg_data.csv"

try:
    # Check if file exists to handle header correctly
    file_exists = os.path.isfile(OUTPUT_FILE)

    # Open the serial port directly
    ser = Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"Successfully connected to {COM_PORT} using raw serial.")
    print(f"Logging full EEG data to {OUTPUT_FILE} (Mode: Append)...")
    print("Press Ctrl+C to stop recording.")

    # Using mode='a' to ensure we don't erase previous data
    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Only write the header if the file is brand new
        if not file_exists:
            writer.writerow([
                "Timestamp", "Signal_Quality", "Attention", "Meditation", 
                "Delta", "Theta", "Low_Alpha", "High_Alpha", 
                "Low_Beta", "High_Beta", "Low_Gamma", "Mid_Gamma"
            ])

        # Initialize all tracking variables to 0
        signal, attention, meditation = 0, 0, 0
        delta, theta, low_alpha, high_alpha = 0, 0, 0, 0
        low_beta, high_beta, low_gamma, mid_gamma = 0, 0, 0, 0

        while True:
            # 1. Sync bytes: Wait for the packet header
            if ser.read(1) == b'\xaa' and ser.read(1) == b'\xaa':
                
                # 2. Payload Length
                plen_byte = ser.read(1)
                if not plen_byte: continue
                plen = plen_byte[0]
                
                if plen > 169: continue 
                
                # 3. Read the actual payload
                payload = ser.read(plen)
                if len(payload) < plen: continue
                
                # 4. Read the Checksum
                chksum_byte = ser.read(1)
                if not chksum_byte: continue
                chksum = chksum_byte[0]
                
                # 5. Verify the Checksum
                calc_chksum = sum(payload) & 0xFF
                calc_chksum = (~calc_chksum) & 0xFF
                
                if calc_chksum == chksum:
                    # 6. Parse the Payload Data
                    i = 0
                    while i < plen:
                        code = payload[i]
                        i += 1
                        
                        # Code 0x02 = Poor Signal
                        if code == 0x02:
                            signal = payload[i]
                            i += 1
                            if signal > 0:
                                # Restored your original warning format
                                print(f"[{time.strftime('%H:%M:%S')}] Signal Warning: {signal}/255 (Adjust headset)")
                        
                        # Code 0x04 = Attention
                        elif code == 0x04:
                            attention = payload[i]
                            i += 1
                            
                        # Code 0x05 = Meditation
                        elif code == 0x05:
                            meditation = payload[i]
                            i += 1
                            
                            # Log the row when meditation updates
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            writer.writerow([
                                timestamp, signal, attention, meditation,
                                delta, theta, low_alpha, high_alpha,
                                low_beta, high_beta, low_gamma, mid_gamma
                            ])
                            file.flush()
                            # Restored your original print format
                            print(f"[{timestamp}] Logged -> Sig:{signal} | Att:{attention} | Med:{meditation} | Beta(L/H): {low_beta}/{high_beta}")
                            
                        # Code 0x83 = ASIC EEG Power (8 frequency bands)
                        elif code == 0x83:
                            vlen = payload[i] 
                            i += 1
                            if vlen == 24:
                                delta = (payload[i] << 16) | (payload[i+1] << 8) | payload[i+2]
                                theta = (payload[i+3] << 16) | (payload[i+4] << 8) | payload[i+5]
                                low_alpha = (payload[i+6] << 16) | (payload[i+7] << 8) | payload[i+8]
                                high_alpha = (payload[i+9] << 16) | (payload[i+10] << 8) | payload[i+11]
                                low_beta = (payload[i+12] << 16) | (payload[i+13] << 8) | payload[i+14]
                                high_beta = (payload[i+15] << 16) | (payload[i+16] << 8) | payload[i+17]
                                low_gamma = (payload[i+18] << 16) | (payload[i+19] << 8) | payload[i+20]
                                mid_gamma = (payload[i+21] << 16) | (payload[i+22] << 8) | payload[i+23]
                                i += 24
                        
                        elif code >= 0x80:
                            vlen = payload[i]
                            i += 1 + vlen
                        else:
                            i += 1

except KeyboardInterrupt:
    print("\nData recording stopped by user.")
except SerialException as e:
    print(f"\nSerial Port Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()