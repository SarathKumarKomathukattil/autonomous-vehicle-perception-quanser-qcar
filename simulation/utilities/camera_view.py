import cv2
import numpy as np
import time
import sys
from io import StringIO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels
from qvl.traffic_light import QLabsTrafficLight

print("\n" + "="*70)
print("🚦 TRAFFIC LIGHT STATE TESTER")
print("Testing ALL possible states to find GREEN")
print("="*70)

qlabs = QuanserInteractiveLabs()
qlabs.open("localhost")
qlabs.destroy_all_spawned_actors()
QLabsRealTime().terminate_all_real_time_models()
time.sleep(1)

print("\n🚗 Spawning QCar...")
qcar = QLabsQCar(qlabs)
qcar.spawn_id(actorNumber=0, location=[-0.15, 3, 0.01], rotation=[0, 0, 300], waitForConfirmation=True)
print("✅ QCar spawned!")

print("\n🚦 Spawning ONE test traffic light...")
light = QLabsTrafficLight(qlabs)
light.spawn(location=[-2.95, 5.6, 0], rotation=[0,0,300], configuration=0)
print("✅ Light spawned!")

QLabsRealTime().start_real_time_model(rtmodels.QCAR)
time.sleep(2)

print("\n" + "="*70)
print("TESTING STATES:")
print("="*70)

# Test states 0-10 to see what each one shows
for state in range(11):
    print(f"\n🔄 Setting light to STATE {state}...")
    light.set_state(state)
    time.sleep(2)  # Wait to see the change
    
    # Get image to show what we see
    success, image_data = qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
    if success and image_data is not None:
        img = np.frombuffer(image_data, dtype=np.uint8).reshape((410, 820, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Add text showing state
        cv2.putText(img, f"STATE {state}", (300, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, "Look at the traffic light color!", (200, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Traffic Light State Test', img)
        
        print(f"   👀 Look at the window - what color is the light?")
        print(f"   Press any key to continue...")
        cv2.waitKey(0)

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print("Which state showed GREEN? Please note the number.")
print("="*70)

cv2.destroyAllWindows()
QLabsRealTime().terminate_all_real_time_models()