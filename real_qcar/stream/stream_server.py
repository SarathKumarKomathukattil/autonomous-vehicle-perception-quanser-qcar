'''laptop_server_camera_feed.py

Simple CSI Camera Feed Display - Server receives frames from QCar and displays them
'''

from pal.utilities.stream import BasicStream
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import time
import numpy as np
import cv2

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001

print("=" * 60)
print("=== CSI Camera Feed Display ===")
print("=" * 60)

# === CREATE SERVER ===
imageData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)

myServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT}',
    agent='S',
    sendBufferSize=controlBufferSize,
    recvBufferSize=imageBufferSize,
    receiveBuffer=imageData,
    nonBlocking=False
)

timeout = Timeout(seconds=5, nanoseconds=0)

print(f"✅ Server started on port {SERVER_PORT}")
print("⏳ Waiting for QCar connection...")

# === WAIT FOR CONNECTION ===
while not myServer.connected:
    myServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)

print("\n✅ QCar Connected!")
print("\nCONTROLS:")
print("  Q = Quit")
print("=" * 60)

frameCount = 0

# === MAIN LOOP ===
try:
    while True:
        # Receive frame
        recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)

        if recvFlag and bytesReceived > 0:
            frameCount += 1
            frame = myServer.receiveBuffer.copy()

            # Display frame with frame counter
            display = frame.copy()
            cv2.putText(display, f'Frame: {frameCount}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('CSI Camera Feed', display)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('\nQuitting...')
                break

            # Send zero control (car stays stopped)
            controlData[0] = 0.0  # throttle
            controlData[1] = 0.0  # steering
            myServer.send(controlData.data)

            # Print status every 60 frames
            if frameCount % 60 == 0:
                print(f'Frame {frameCount} received')

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Send stop command
    controlData[0] = 0.0
    controlData[1] = 0.0
    if myServer.connected:
        myServer.send(controlData.data)
    myServer.terminate()
    cv2.destroyAllWindows()
    print('Done!')