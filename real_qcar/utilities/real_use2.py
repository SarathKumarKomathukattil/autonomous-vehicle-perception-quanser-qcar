from pal.products.traffic_light import TrafficLight
import time

# ===== CONFIGURATION =====
light_ip = '192.168.2.14'  # Your traffic light IP
RED_TIME = 8    # seconds
YELLOW_TIME = 2   # seconds
GREEN_TIME = 4   # seconds
# =========================

print("Connecting to traffic light...")
light = TrafficLight(light_ip)

# Check connection
status = light.status()
print(f"✓ Connected! Status: {status}\n")

# IMPORTANT: Turn off first (in case previous cycle is running)
print("Ensuring light is off...")
light.off()
time.sleep(2)

# Start cycling
print(f"Starting cycle: {RED_TIME}s RED → {YELLOW_TIME}s YELLOW → {GREEN_TIME}s GREEN")
light.timed(RED_TIME, YELLOW_TIME, GREEN_TIME)
print("✓ Traffic light is now cycling!\n")
print("Watch the traffic light cycle automatically.")
print("Press Ctrl+C to stop\n")

try:
    # Keep program running and show current status
    while True:
        status = light.status()
        status_names = ['OFF', 'RED', 'YELLOW', 'GREEN']
        current = status_names[int(status)]
        print(f"Current light: {current}", end='\r')
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print("\n\nStopping traffic light...")
    light.off()
    print("✓ Done!")